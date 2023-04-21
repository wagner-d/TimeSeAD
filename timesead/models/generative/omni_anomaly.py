import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from ..common import RNN, DenseVAEEncoder, AnomalyDetector
from ..layers import PlanarFlow
from ...models import BaseModel
from ...optim.loss import Loss


class KalmanFilter(torch.nn.Module):
    def __init__(self, state_dim: int, observation_dim: int):
        super(KalmanFilter, self).__init__()

        self.state_dim = state_dim
        self.observation_dim = observation_dim

        self.F = torch.nn.Parameter(torch.eye(state_dim, dtype=torch.float))
        self.state_cov = torch.nn.Parameter(torch.ones(state_dim, dtype=torch.float))

        H = torch.zeros(observation_dim, state_dim, dtype=torch.float)
        H.diagonal().add_(1)
        self.H = torch.nn.Parameter(H)
        self.obs_cov = torch.nn.Parameter(torch.ones(observation_dim, dtype=torch.float))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (T, L, B, D)
        T, B, D = observations.shape

        assert D == self.observation_dim

        s_cov = F.softplus(self.state_cov)
        obs_cov = F.softplus(self.obs_cov)

        log_likelihood = 0
        state_mean = torch.zeros(B, self.state_dim, dtype=observations.dtype, device=observations.device)
        state_cov = torch.eye(self.state_dim, dtype=observations.dtype, device=observations.device)

        for t in range(T):
            observation = observations[t]
            state_upd_mean = state_mean @ self.F.T
            state_update_cov = self.F @ state_cov @ self.F.T + s_cov.diag_embed()

            innovation = observation - state_upd_mean @ self.H.T
            innovation_cov = self.H @ state_update_cov @ self.H.T + obs_cov.diag_embed()

            expanded_innovation_cov = innovation_cov.view(1, self.observation_dim, self.observation_dim)
            expanded_innovation_cov = expanded_innovation_cov.expand(B, self.observation_dim, self.observation_dim)
            solve_mean = torch.linalg.solve(expanded_innovation_cov, innovation)
            solve_cov = torch.linalg.solve(innovation_cov, self.H)

            PH = state_update_cov @ self.H.T

            state_mean = state_upd_mean + solve_mean @ PH.T
            state_cov = state_update_cov - PH @ solve_cov @ state_update_cov

            log_likelihood = log_likelihood - 0.5 * (torch.inner(innovation, solve_mean) + torch.logdet(innovation_cov)
                                                     + self.observation_dim * math.log(2 * math.pi))

        return log_likelihood


class OmniAnomaly(BaseModel):
    def __init__(self, input_dim: int, latent_dim: int = 3, rnn_hidden_dims: Sequence[int] = (500,),
                 dense_hidden_dims: Sequence[int] = (500, 500), nf_layers: int = 20):
        super(OmniAnomaly, self).__init__()

        self.latent_dim = latent_dim

        self.prior = KalmanFilter(latent_dim, latent_dim)

        self.enc_rnn = RNN('gru', 's2s', input_dim, rnn_hidden_dims)
        self.encoder_vae = DenseVAEEncoder(rnn_hidden_dims[-1] + latent_dim, dense_hidden_dims, latent_dim)
        self.latent_nf = PlanarFlow(latent_dim, num_layers=nf_layers)

        self.decoder_rnn = RNN('gru', 's2s', latent_dim, rnn_hidden_dims)
        self.decoder_vae = DenseVAEEncoder(rnn_hidden_dims[-1], dense_hidden_dims, input_dim)

    def forward(self, inputs: Tuple[torch.Tensor], num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                  torch.Tensor, torch.Tensor,
                                                                                  torch.Tensor, torch.nn.Module,
                                                                                  torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        x, = inputs
        T, B, D = x.shape

        # Use RNN to encode the input
        hidden = self.enc_rnn(x)
        # Add sample dimension
        hidden = hidden.unsqueeze(1).expand(T, num_samples, B, hidden.shape[-1])

        normal_sample = torch.randn((T+1, num_samples, B, self.latent_dim), dtype=x.dtype, device=x.device)
        z_sample = [normal_sample[0]]
        z_mean, z_std = [], []
        for t in range(T):
            z_t_mean, z_t_std = self.encoder_vae(torch.cat([hidden[t], z_sample[t]], dim=-1))
            z_mean.append(z_t_mean)
            z_std.append(z_t_std)
            # From the paper it might seem that the normalizing flow is used during the sequential computation of z,
            # but in the code this is not the case. We choose to implement it like in the code
            z_sample.append(z_t_std * normal_sample[t+1] + z_t_mean)
        z_mean = torch.stack(z_mean, dim=0)
        z_std = torch.stack(z_std, dim=0)
        z_sample = torch.stack(z_sample[1:], dim=0)

        # Transform samples using a planar normalizing flow
        z_sample_transformed, z_log_det = self.latent_nf(z_sample)

        # Decode. Collapse sample and batch dimension for RNN
        z_sample_transformed = z_sample_transformed.view(T, -1, self.latent_dim)
        dec_hidden = self.decoder_rnn(z_sample_transformed)
        # Restore sample dimension
        dec_hidden = dec_hidden.view(T, num_samples, B, dec_hidden.shape[-1])
        x_rec_mean, x_rec_std = self.decoder_vae(dec_hidden)

        return z_std, z_mean, z_sample, z_sample_transformed, z_log_det, self.prior, x_rec_mean, x_rec_std


class OmniAnomalyLoss(Loss):
    def __init__(self):
        super(OmniAnomalyLoss, self).__init__(reduction='mean')

    def forward(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                         torch.nn.Module, torch.Tensor, torch.Tensor],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        z_std, z_mean, z_sample, z_sample_transformed, z_log_det, prior, x_rec_mean, x_rec_std = predictions
        actual_x, = targets

        # L = sample dimension
        T, L, B, D = x_rec_mean.shape

        nll_output = F.gaussian_nll_loss(x_rec_mean, actual_x.unsqueeze(1), x_rec_std**2, reduction='sum') / (B * L)

        # We cannot compute KL(q(z|x) || p(z)) exactly anymore because q is non-gaussian, so we have to use
        # MC approximation to estimate E_{z~q(z|x)}[log(q(z|x)) - log(p(z))]

        # This is p(z), i.e., the prior likelihood of Z. The paper assumes a linear gaussian state space model
        # aka. a Kalman filter
        ll_prior = torch.mean(prior(z_sample_transformed))

        # log(q(zx)) can be computed from the original sample's density and the NF Jacobian's log det.
        nll_approx = F.gaussian_nll_loss(z_mean, z_sample, z_std**2, reduction='sum') / (B * L)
        nll_approx = nll_approx - torch.sum(z_log_det) / (B * L)

        return nll_output - ll_prior - nll_approx


class OmniAnomalyDetector(AnomalyDetector):
    def __init__(self, model: OmniAnomaly, num_mc_samples: int = 1024):
        """
        We decided not to include the reconstruction step from the paper here, since we don't have missing data.

        :param model:
        :param num_mc_samples:
        """
        super(OmniAnomalyDetector, self).__init__()

        self.model = model
        self.num_mc_samples = num_mc_samples

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # We compute not only the log likelihood of the data, but the entire ELBO
        # x: (T, B, D)
        # output (B,)
        x, = inputs

        with torch.no_grad():
            res = self.model(inputs, num_samples=self.num_mc_samples)

        z_std, z_mean, z_sample, z_sample_transformed, z_log_det, prior, x_rec_mean, x_rec_std = res

        # Compute MC approximation of Log likelihood
        nll_output = torch.sum(F.gaussian_nll_loss(x_rec_mean[-1], x[-1].unsqueeze(0),
                                                   x_rec_std[-1]**2, reduction='none'), dim=(0, -1))
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
