from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from .lstm_vae import RNNVAEGaussianEncoder
from ..common import RNN, DenseVAEEncoder, VAE, VAELoss, AnomalyDetector
from ...models import BaseModel


class RNNVAECategoricalEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, rnn_type: str = 'lstm', rnn_hidden_dims: List[int] = [60], categories: int = 10,
                 bidirectional: bool = False, mode: str = 's2s'):
        super(RNNVAECategoricalEncoder, self).__init__()

        self.rnn = RNN(rnn_type, mode, input_dim, rnn_hidden_dims, bidirectional=bidirectional)
        out_hidden_size = 2 * rnn_hidden_dims[-1] if bidirectional else rnn_hidden_dims[-1]
        self.linear = torch.nn.Linear(out_hidden_size, categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        rnn_out = self.rnn(x)
        logits = self.linear(rnn_out)

        return logits


class GRUGMMVAE(BaseModel):
    def __init__(self, input_dim: int, gru_hidden_dims: List[int] = [60], latent_dim: int = 8, gmm_components: int = 2):
        """
        Guo2018 (more or less)

        :param input_dim:
        :param gru_hidden_dims:
        :param latent_dim:
        """
        super(GRUGMMVAE, self).__init__()

        self.latent_dim = latent_dim
        self.gmm_components = gmm_components

        self.encoder_rnn = RNN('gru', 's2s', input_dim, gru_hidden_dims)
        self.encoder_component = torch.nn.Linear(gru_hidden_dims[-1], gmm_components)

        encoder_normal = DenseVAEEncoder(gru_hidden_dims[-1] + gmm_components, gru_hidden_dims, latent_dim)
        decoder = RNNVAEGaussianEncoder(latent_dim, 'gru', gru_hidden_dims[::-1], input_dim, logvar_out=False)
        self.vae = VAE(encoder_normal, decoder, logvar_out=False)

        self.prior_means = torch.nn.Parameter(torch.rand(gmm_components, latent_dim))
        self.prior_std = torch.nn.Parameter(torch.rand(gmm_components, latent_dim))

        self.softplus = torch.nn.Softplus()

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mean = self.prior_means.view(1, 1, *self.prior_means.shape)
        mean = mean.expand(seq_len, batch_size, -1, -1)

        std = self.prior_std.view(1, 1, *self.prior_std.shape)
        std = self.softplus(std)
        std = std.expand(seq_len, batch_size, -1, -1)

        return mean, std

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (T, B, D)
        x, = inputs
        T, B, D = x.shape

        means_z_prior, stds_z_prior = self.get_prior(B, T)

        # Run RNN on the input
        hidden = self.encoder_rnn(x)
        # Get the categorical distribution for the mixture components
        component_logits = self.encoder_component(hidden)

        # Run the VAE for each component
        means_z = []
        stds_z = []
        means_x = []
        stds_x = []
        one_hot = torch.zeros(self.gmm_components, dtype=torch.float, device=x.device)
        for k in range(self.gmm_components):
            one_hot[k] = 1
            mean_z, std_z, mean_x, std_x = self.vae(torch.cat([hidden, one_hot.expand(T, B, -1)], dim=-1))
            one_hot[k] = 0

            means_z.append(mean_z)
            stds_z.append(std_z)
            means_x.append(mean_x)
            stds_x.append(std_x)

        means_z = torch.stack(means_z, dim=-2)
        stds_z = torch.stack(stds_z, dim=-2)
        means_x = torch.stack(means_x, dim=-2)
        stds_x = torch.stack(stds_x, dim=-2)

        return means_z, stds_z, means_x, stds_x, means_z_prior, stds_z_prior, component_logits


class GMMVAELoss(VAELoss):
    def __init__(self):
        super(GMMVAELoss, self).__init__(reduction='none', logvar_out=False)

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        means_z, stds_z, means_x, stds_x, means_z_prior, stds_z_prior, component_logits = predictions
        actual_x, = targets

        # Computing the KL divergence between a categorical distribution and a uniform prior reduces to computing its
        # negative entropy + constants (log(k))
        cat_dist = torch.distributions.Categorical(logits=component_logits)
        cat_probs = cat_dist.probs
        cat_entropy = torch.mean(cat_dist.entropy())

        # means_z: (T, B, k, latent), means_x: (T, B, k, D)
        T, B, k, D = means_x.shape
        normal_losses = super(GMMVAELoss, self).forward(
            (means_z, stds_z, means_x, stds_x, means_z_prior, stds_z_prior),
            (actual_x.unsqueeze(-2),)
        )
        loss = -cat_entropy + torch.dot(cat_probs.flatten(), normal_losses.flatten()) / (T * B)

        return loss


class GMMVAEAnomalyDetector(AnomalyDetector):
    def __init__(self, model: GRUGMMVAE, num_mc_samples: int = 1):
        """
        Use sampled log likelihood of data

        :param model:
        :param num_mc_samples:
        """
        super(GMMVAEAnomalyDetector, self).__init__()

        self.model = model
        self.num_mc_samples = num_mc_samples

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # We compute not only the log likelihood of the data, but the entire ELBO
        # x: (T, B, D)
        # output (B,)
        x, = inputs

        nll_output = 0
        with torch.no_grad():
            hidden = self.model.encoder_rnn(x)
            component_logits = self.model.encoder_component(hidden)
            component_dist = torch.distributions.OneHotCategorical(logits=component_logits)
            for _ in range(self.num_mc_samples):
                cat_sample = component_dist.sample()
                x_cat = torch.cat([hidden, cat_sample], dim=-1)
                z_mean, z_std, x_mean, x_std = self.model.vae(x_cat, force_sample=True)

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
