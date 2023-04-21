import math
from typing import Callable, Union, Tuple, List, Type

import torch
from torch.nn import functional as F

from ..common import AnomalyDetector, TCN
from ...models import BaseModel
from ...utils import torch_utils


class TCNAE(BaseModel):
    """
    A class used to represent the Temporal Convolutional Autoencoder (TCN-AE).
    Loss for this is logcosh
    """

    def __init__(self, input_dimension: int,
                 dilations: List[int] = (1, 2, 4, 8, 16),
                 nb_filters: Union[int, List[int]] = 20,
                 kernel_size: int = 20,
                 nb_stacks: int = 1,
                 padding: str = 'same',
                 dropout_rate: float = 0.00,
                 filters_conv1d: int = 8,
                 activation_conv1d: Union[str, Callable] = 'linear',
                 latent_sample_rate: int = 42,
                 pooler: Type[torch.nn.Module] = torch.nn.AvgPool1d):
        """
        Parameters
        ----------
        ts_dimension : int
            The dimension of the time series (default is 1)
        dilations : tuple
            The dilation rates used in the TCN-AE model (default is (1, 2, 4, 8, 16))
        nb_filters : int
            The number of filters used in the dilated convolutional layers. All dilated conv. layers use the same number of filters (default is 20)
        """

        super().__init__()

        self.tcn_enc = TCN(input_dimension, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=True, dropout_rate=dropout_rate,
                           return_sequences=True)

        self.conv1d = torch.nn.Conv1d(nb_filters, filters_conv1d, kernel_size=1, padding=padding)

        if isinstance(activation_conv1d, str):
            activation_conv1d = torch_utils.activations[activation_conv1d]
        self.activation = activation_conv1d

        self.pooler = pooler(kernel_size=latent_sample_rate)

        self.tcn_dec = TCN(filters_conv1d, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=True, dropout_rate=dropout_rate,
                           return_sequences=True)

        self.linear = torch.nn.Conv1d(nb_filters, input_dimension, kernel_size=1, padding=padding)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """

        :param inputs: Tuple with single Tensor of shape (B, T, D)
        :return:
        """
        # Transpose the input to the required format (B, D, T)
        x, = inputs
        x = x.transpose(1, 2)

        # Put signal through TCN. Output-shape: (B, nb_filters, T)
        x = self.tcn_enc(x)
        # Now, adjust the number of channels...
        x = self.conv1d(x)
        x = self.activation(x)

        # Do some average (max) pooling to get a compressed representation of the time series
        # (e.g. a sequence of length 8)
        seq_len = x.shape[-1]
        x = self.pooler(x)
        # x = self.activation(x)

        # Now we should have a short sequence, which we will upsample again and
        # then try to reconstruct the original series
        x = F.interpolate(x, seq_len, mode='nearest')
        x = self.tcn_dec(x)
        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        x = self.linear(x)

        # Put output dimensions in the correct order again
        x = x.transpose(1, 2)

        return x


class TCNAEAnomalyDetector(AnomalyDetector):
    def __init__(self, model: TCNAE, offline_window_size: int = 128):
        super(TCNAEAnomalyDetector, self).__init__()

        self.model = model
        self.offline_window_size = offline_window_size

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        errors = []
        mean = 0
        total = 0

        # Compute mean and covariance over the entire validation dataset
        for i, (b_inputs, b_targets) in enumerate(dataset):
            b_inputs = tuple(b_inp.to(torch_utils.get_device(self.model)) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(torch_utils.get_device(self.model)) for b_tar in b_targets)
            with torch.no_grad():
                pred = self.model(b_inputs)

            # Data points and predictions have shape (B, T, D)
            target, = b_targets

            error = target - pred
            error = error.view(error.shape[0], -1)
            errors.append(error)

            mean += torch.sum(error, dim=0)
            total += error.shape[0]

        mean /= total

        errors = torch.cat(errors, dim=0)
        errors -= mean
        errors /= math.sqrt(total - 1)
        cov = torch.matmul(errors.T, errors)

        # Add a small epsilon to the diagonal of the matrix to make it non-singular
        epsilon = 1e-5
        cov.diagonal().add_(epsilon)

        # This construction ensures that the resulting precision matrix is pos. semi-definite, even if the condition
        # number of the cov matrix is large
        tries = 3
        for i in range(tries):
            try:
                cholesky = torch.linalg.cholesky(cov)
                break
            except RuntimeError as e:
                if i == tries - 1:
                    raise e

                cov.diagonal().add_(10 * epsilon - epsilon)
                epsilon *= 10

        precision = cov
        torch.cholesky_inverse(cholesky, out=precision)

        self.register_buffer('mean', mean)
        # self.register_buffer('precision', precision)
        self.register_buffer('precision', precision)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T, D), output of shape (B,)
        with torch.no_grad():
            prediction = self.model(inputs)

        input, = inputs
        error = (input - prediction).view(input.shape[0], -1)

        error -= self.mean

        result = F.bilinear(error, error, self.precision.unsqueeze(0))
        return result.squeeze(-1)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # input shape (B, T, D)
        # return shape (B, (T-window_size+1))
        x,  = inputs
        B, T, D = x.shape
        with torch.no_grad():
            reconstruction = self.model(inputs).cpu()

        error = reconstruction - x.cpu()
        # Compute mean and covariance
        mean = torch.zeros(B, self.offline_window_size*D)
        for i in range(self.offline_window_size, T+1):
            end = i
            start = end - self.offline_window_size
            window = error[:, start:end, :]
            mean += window.reshape(-1, self.offline_window_size * D) / (T - self.offline_window_size)

        cov = torch.zeros(B, self.offline_window_size*D, self.offline_window_size*D)
        for i in range(self.offline_window_size, T+1):
            end = i
            start = end - self.offline_window_size
            window = error[:, start:end, :]
            window = window.reshape(-1, self.offline_window_size * D)
            window = window - mean
            cov += torch.matmul(window.unsqueeze(-1), window.unsqueeze(-2)) / (T - self.offline_window_size - 1)

        precision = torch.inverse(cov)

        distances = []
        for i in range(self.offline_window_size, T+1):
            end = i
            start = end - self.offline_window_size
            window = error[:, start:end, :]
            window = window.reshape(-1, self.offline_window_size * D)
            window = window - mean
            distances.append(F.bilinear(window, window, precision.unsqueeze(0)).squeeze(-1))

        # Note that we don't have a score for the first (window_size - 1) points
        return torch.tensor(distances)

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[:, -1]