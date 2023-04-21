from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..common import MLP, DenseVAEEncoder, VAELoss, AnomalyDetector
from ..common.vae import sample_normal, normal_normal_kl
from ...models import BaseModel


class SISVAE(BaseModel):
    def __init__(self, input_dim: int, rnn_hidden_dim: int = 200, latent_dim: int = 40,
                 x_hidden_dims: List[int] = [100], z_hidden_dims: List[int] = [100],
                 enc_hidden_dims: List[int] = [100], dec_hidden_dims: List[int] = [100],
                 prior_hidden_dims: List[int] = [100]):
        """
        Li2021, ist aber im Prinzip nur Chung2015 mit einem extra loss term

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        """
        super(SISVAE, self).__init__()

        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.x_embed = MLP(input_dim, x_hidden_dims[:-1], x_hidden_dims[-1], activation=torch.nn.ReLU(),
                           activation_after_last_layer=True)
        self.z_embed = MLP(latent_dim, z_hidden_dims[:-1], z_hidden_dims[-1], activation=torch.nn.ReLU(),
                           activation_after_last_layer=True)

        self.encoder = DenseVAEEncoder(x_hidden_dims[-1] + rnn_hidden_dim, enc_hidden_dims, latent_dim)
        self.decoder = DenseVAEEncoder(z_hidden_dims[-1] + rnn_hidden_dim, dec_hidden_dims, input_dim)
        self.prior_decoder = DenseVAEEncoder(rnn_hidden_dim, prior_hidden_dims, latent_dim)

        self.rnn_cell = torch.nn.GRUCell(x_hidden_dims[-1] + z_hidden_dims[-1], rnn_hidden_dim)

        self.softplus = torch.nn.Softplus()

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (T, B, D)
        x, = inputs
        # T, B, D = x.shape

        # First compute an embedding for x
        # (T, B, hidden_x)
        x = self.x_embed(x)

        hidden = [torch.zeros(x.shape[1], self.rnn_hidden_dim, dtype=x.dtype, device=x.device)]
        z_mean = []
        z_std = []
        z_sample = []
        for t in range(x.shape[0]):
            z_mean_t, z_std_t = self.encoder(torch.cat([x[t], hidden[t]], dim=-1))
            z_sample_t = sample_normal(z_mean_t, z_std_t, log_var=False)
            z_sample_t = self.z_embed(z_sample_t)

            hidden.append(self.rnn_cell(torch.cat([x[t], z_sample_t], dim=-1), hidden[t]))
            z_mean.append(z_mean_t)
            z_std.append(z_std_t)
            z_sample.append(z_sample_t)

        hidden = torch.stack(hidden[:-1], dim=0)
        z_mean = torch.stack(z_mean, dim=0)
        z_std = self.softplus(torch.stack(z_std, dim=0))
        z_sample = torch.stack(z_sample, dim=0)

        prior_mean, prior_std = self.prior_decoder(hidden)
        prior_std = self.softplus(prior_std)

        x_mean, x_std = self.decoder(torch.cat([z_sample, hidden], dim=-1))
        x_std = self.softplus(x_std)

        return z_mean, z_std, x_mean, x_std, prior_mean, prior_std


class SISVAELossWithGeneratedPrior(VAELoss):
    def __init__(self, smooth_weight: float = 0.5):
        super(SISVAELossWithGeneratedPrior, self).__init__(reduction='mean', logvar_out=False)

        self.smooth_weight = smooth_weight

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        z_mean, z_std, x_mean, x_std, z_prior_mean, z_prior_std = predictions

        neg_elbo = super(SISVAELossWithGeneratedPrior, self).forward(predictions, targets)
        smooth_loss = torch.mean(normal_normal_kl(x_mean[:-1], x_std[:-1], x_mean[1:], x_std[1:], log_var=False))

        # Combine
        return neg_elbo + self.smooth_weight * smooth_loss


class SISVAEAnomalyDetector(AnomalyDetector):
    def __init__(self, model: SISVAE, num_mc_samples: int = 128):
        """
        We decided not to include the reconstruction step from the paper here, since we don't have missing data.

        :param model:
        :param num_mc_samples:
        """
        super(SISVAEAnomalyDetector, self).__init__()

        self.model = model
        self.num_mc_samples = num_mc_samples

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # We compute not only the log likelihood of the data, but the entire ELBO
        # x: (T, B, D)
        # output (B,)
        x, = inputs

        nll_output = 0
        for _ in range(self.num_mc_samples):
            with torch.no_grad():
                res = self.model(inputs)

            z_mean, z_std, x_mean, x_std, prior_mean, prior_std = res

            # Compute MC approximation of Log likelihood
            nll_output += torch.sum(F.gaussian_nll_loss(x_mean[-1], x[-1], x_std[-1]**2, reduction='none'), dim=-1)
        nll_output /= self.num_mc_samples

        return nll_output

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[-1]
