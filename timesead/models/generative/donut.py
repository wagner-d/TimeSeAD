from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..common import DenseVAEEncoder, VAE, VAELoss, AnomalyDetector
from ...models import BaseModel


class Donut(BaseModel):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [100, 100], latent_dim: int = 20,
                 mask_prob: float = 0.01):
        """
        Xu2018

        :param input_dim: Should be window_size * features
        :param hidden_dims:
        :param latent_dim:
        """
        super(Donut, self).__init__()

        self.latent_dim = latent_dim
        self.mask_prob = mask_prob

        encoder = DenseVAEEncoder(input_dim, hidden_dims, latent_dim)
        decoder = DenseVAEEncoder(latent_dim, hidden_dims[::-1], input_dim)
        self.vae = VAE(encoder, decoder, logvar_out=False)

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (B, T, D)
        x, = inputs
        B, T, D = x.shape

        if self.training:
            # Randomly mask some inputs
            mask = torch.empty_like(x)
            mask.bernoulli_(1 - self.mask_prob)
            x = x * mask
        else:
            mask = None

        # Run the VAE
        x = x.view(x.shape[0], -1)
        mean_z, std_z, mean_x, std_x, sample_z = self.vae(x, return_latent_sample=True)

        # Reshape the outputs
        mean_x = mean_x.view(B, T, D)
        std_x = std_x.view(B, T, D)

        return mean_z, std_z, mean_x, std_x, sample_z, mask


class MaskedVAELoss(VAELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MaskedVAELoss, self).__init__(size_average, reduce, reduction, logvar_out=False)

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        mean_z, std_z, mean_x, std_x, sample_z, mask = predictions
        actual_x, = targets

        if mask is None:
            mean_z = mean_z.unsqueeze(1)
            std_z = std_z.unsqueeze(1)
            return super(MaskedVAELoss, self).forward((mean_z, std_z, mean_x, std_x), (actual_x,), *args, **kwargs)

        # If the loss is masked, one of the terms in the kl loss is weighted, so we can't compute it exactly
        # anymore and have to use a MC approximation like for the output likelihood
        nll_output = torch.sum(mask * F.gaussian_nll_loss(mean_x, actual_x, std_x**2, reduction='none'), dim=-1)

        # This is p(z), i.e., the prior likelihood of Z. The paper assumes p(z) = N(z| 0, I), we drop constants
        beta = torch.mean(mask, dim=(1, 2)).unsqueeze(-1)
        nll_prior = beta * 0.5 * torch.sum(sample_z * sample_z, dim=-1, keepdim=True)

        nll_approx = torch.sum(F.gaussian_nll_loss(mean_z, sample_z, std_z**2, reduction='none'), dim=-1, keepdim=True)

        final_loss = nll_output + nll_prior - nll_approx

        if self.reduction == 'none':
            return final_loss
        elif self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)


class DonutAnomalyDetector(AnomalyDetector):
    def __init__(self, model: Donut, num_mc_samples: int = 1024):
        """
        We decided not to include the reconstruction step from the paper here, since we don't have missing data.

        :param model:
        :param num_mc_samples:
        """
        super(DonutAnomalyDetector, self).__init__()

        self.model = model
        self.num_mc_samples = num_mc_samples

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # We compute not only the log likelihood of the data, but the entire ELBO
        # x: (B, T, D)
        # output (B,)
        x, = inputs
        B, T, D = x.shape

        with torch.no_grad():
            x_vae = x.view(x.shape[0], -1)
            res = self.model.vae(x_vae, return_latent_sample=False, num_samples=self.num_mc_samples)

        z_mu, z_std, x_dec_mean, x_dec_std = res
        # Reshape the outputs
        x_dec_mean = x_dec_mean.view(self.num_mc_samples, B, T, D)
        x_dec_std = x_dec_std.view(self.num_mc_samples, B, T, D)

        # Compute MC approximation of Log likelihood
        nll_output = torch.sum(F.gaussian_nll_loss(x_dec_mean[:, :, -1, :], x[:, -1, :].unsqueeze(0),
                                                   x_dec_std[:, :, -1, :]**2, reduction='none'), dim=(0, 2))
        nll_output /= self.num_mc_samples

        return nll_output

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[:, -1]
