from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVR

from ..common import RNN, VAE, AnomalyDetector, VAELoss
from ...models import BaseModel
from ...utils import torch_utils


class RNNVAEGaussianEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, rnn_type: str = 'lstm', rnn_hidden_dims: List[int] = [60], latent_dim: int = 10,
                 bidirectional: bool = False, mode: str = 's2s', logvar_out: bool = True):
        super(RNNVAEGaussianEncoder, self).__init__()

        self.logvar = logvar_out

        self.rnn = RNN(rnn_type, mode, input_dim, rnn_hidden_dims, bidirectional=bidirectional)
        out_hidden_size = 2 * rnn_hidden_dims[-1] if bidirectional else rnn_hidden_dims[-1]
        self.linear = torch.nn.Linear(out_hidden_size, 2 * latent_dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        rnn_out = self.rnn(x)
        mean, std_or_logvar = self.linear(rnn_out).tensor_split(2, dim=-1)

        if not self.logvar:
            std_or_logvar = self.softplus(std_or_logvar)

        return mean, std_or_logvar


class LSTMVAE(BaseModel):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60], latent_dim: int = 20):
        """
        Base LSTMVAE

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        """
        super(LSTMVAE, self).__init__()

        self.latent_dim = latent_dim

        encoder = RNNVAEGaussianEncoder(input_dim, rnn_type='lstm', rnn_hidden_dims=lstm_hidden_dims,
                                        latent_dim=latent_dim, logvar_out=True)
        decoder = RNNVAEGaussianEncoder(latent_dim, rnn_type='lstm', rnn_hidden_dims=lstm_hidden_dims[::-1],
                                        latent_dim=input_dim, logvar_out=False)
        self.vae = VAE(encoder, decoder, logvar_out=True)

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (T, B, D)
        x, = inputs
        T, B, D = x.shape

        mean_z_prior, logvar_z_prior = self.get_prior(B, T)

        # Run the VAE
        mean_z, logvar_z, mean_x, std_x = self.vae(x)

        return mean_z, logvar_z, mean_x, std_x, mean_z_prior, logvar_z_prior


class LSTMVAESoelch(LSTMVAE):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60], latent_dim: int = 20,
                 prior_hidden_dim: int = 40):
        """
        Sölch2016

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        :param prior_hidden_dim:
        """
        super(LSTMVAESoelch, self).__init__(input_dim, lstm_hidden_dims, latent_dim)

        self.prior_hidden_dim = prior_hidden_dim

        self.prior_rnn = torch.nn.LSTMCell(latent_dim, prior_hidden_dim)
        self.prior_linear = torch.nn.Linear(prior_hidden_dim, latent_dim)

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mean_z_prior = []
        hidden = None
        # Init the first prior
        mean_z_prior.append(self.prior_linear(torch.zeros(batch_size, self.prior_hidden_dim,
                                                          dtype=self.prior_linear.weight.dtype,
                                                          device=self.prior_linear.weight.device)))
        for t in range(1, seq_len):
            hidden = self.prior_rnn(mean_z_prior[t - 1], hidden)
            mean_z_prior.append(self.prior_linear(hidden[0]))

        mean_z_prior = torch.stack(mean_z_prior, dim=0)

        return mean_z_prior, None


class LSTMVAEPark(LSTMVAE):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60], latent_dim: int = 20,
                 noise_std: float = 0.1):
        """
        Park2018

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        :param noise_std:
        """
        super(LSTMVAEPark, self).__init__(input_dim, lstm_hidden_dims, latent_dim)

        self.noise_std = noise_std

        self.prior_means = torch.nn.Parameter(torch.zeros(2, latent_dim))

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        alphas = torch.linspace(0, 1, steps=seq_len, dtype=self.prior_means.dtype, device=self.prior_means.device)
        alphas = alphas.view(-1, 1, 1)
        mean_z_prior = alphas * self.prior_means[0].view(1, 1, -1) + (1 - alphas) * self.prior_means[1].view(1, 1, -1)
        mean_z_prior = mean_z_prior.expand(seq_len, batch_size, self.latent_dim)

        return mean_z_prior, None

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        x_orig, = inputs
        if self.training:
            # Corrupt input data
            x = torch.randn_like(x_orig)
            x *= self.noise_std
            x += x_orig
        else:
            x = x_orig

        return super(LSTMVAEPark, self).forward((x,))


class VAEAnomalyDetectorSoelch(AnomalyDetector):
    def __init__(self, model: LSTMVAESoelch):
        super(VAEAnomalyDetectorSoelch, self).__init__()

        self.model = model
        self.loss = VAELoss(reduction='none', logvar_out=True)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # We compute not only the log likelihood of the data, but the entire ELBO
        # x: (T, B, D)
        # output (B,)
        x, = inputs
        with torch.no_grad():
            res = self.model(inputs)

        loss = self.loss(res, inputs)
        return loss[-1]

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[-1]


class VAEAnomalyDetectorPark(AnomalyDetector):
    def __init__(self, model: LSTMVAEPark, num_mc_samples: int = 1):
        """
        Use sampled log likelihood of data + some thresholding mechanism

        :param model:
        :param num_mc_samples:
        """
        super(VAEAnomalyDetectorPark, self).__init__()

        self.model = model
        self.num_mc_samples = num_mc_samples
        self.svr = SVR(kernel='rbf')

    def compute_threshold(self, z: torch.Tensor):
        # z: (B, D)
        z_numpy = z.cpu().numpy()

        threshold = self.svr.predict(z_numpy)
        return torch.from_numpy(threshold).to(z.device)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (T, B, D)
        # output (B,)
        x, = inputs

        nll_output = 0
        z_sample = 0
        for _ in range(self.num_mc_samples):
            with torch.no_grad():
                res = self.model.vae(x, return_latent_sample=True, force_sample=True)

            z_mean, z_std, x_mean, x_std, z = res

            # Compute MC approximation of Log likelihood
            nll_output += torch.sum(F.gaussian_nll_loss(x_mean[-1], x[-1], x_std[-1] ** 2, reduction='none'), dim=-1)
            z_sample += z[-1]

        nll_output /= self.num_mc_samples
        z_sample /= self.num_mc_samples

        threshold = self.compute_threshold(z_sample)

        return nll_output - threshold

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        # Train an SVR on the output of the validation set
        z_train = []
        nll_train = []

        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(torch_utils.get_device(self.model)) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(torch_utils.get_device(self.model)) for b_tar in b_targets)

            x, = b_inputs
            with torch.no_grad():
                pred = self.model.vae(x, return_latent_sample=True)

            z_mean, z_logvar, x_mean, x_std, z = pred
            x_actual, = b_targets

            nll_x = F.gaussian_nll_loss(x_mean[-1], x_actual[-1], x_std[-1] ** 2, reduction='none').sum(-1)

            z_train.append(z[-1].cpu().numpy())
            nll_train.append(nll_x.cpu().numpy())

        # (N, D)
        z_train = np.concatenate(z_train, axis=0)
        # (N,)
        nll_train = np.concatenate(nll_train, axis=0)

        self.svr.fit(z_train, nll_train)

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[-1]
