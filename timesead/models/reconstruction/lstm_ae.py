import abc
import functools
from typing import Tuple, Union, Callable, List, Type

import torch
import torch.nn.functional as F

from ..common import AE, AnomalyDetector
from ..common import RNN
from ...models import BaseModel
from ...utils import torch_utils
from ...utils.utils import getitem


class LSTMAEDecoder(torch.nn.Module, abc.ABC):
    def __init__(self, enc_hidden_dimension: int, hidden_dimensions: List[int], output_dimension: int):
        super().__init__()

    @abc.abstractmethod
    def forward(self, initial_hidden: torch.Tensor, seq_len: int, x: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class LSTMAEDecoderSimple(LSTMAEDecoder):
    """
    Reconstruct the time series using a LSTM decoder, starting with an initial hidden state from the encoder
    that is used as input to every timestep of the decoder
    This corresponds to Mirza 2018
    """
    def __init__(self, enc_hidden_dimension: int, hidden_dimensions: List[int], output_dimension: int):
        super().__init__(enc_hidden_dimension, hidden_dimensions, output_dimension)

        self.lstm = RNN(layer_type='LSTM', input_dimension=enc_hidden_dimension, model='s2s',
                        hidden_dimensions=hidden_dimensions + [output_dimension])

    def forward(self, initial_hidden: List[torch.Tensor], seq_len: int, x: torch.Tensor = None) -> torch.Tensor:
        """

        :param initial_hidden: list (length hidden_layers) of tensors of shape (B, D)
        :param seq_len: int that determines the length of the produced sequence
        :param x: The ground truth sequence that should be reconstructed as a tensor of shape (T, B, D).
         This will be fed into the LSTM during training instead of the output from the previous step.
        :return: Tensor of shape (T, B, D)
        """
        # Repeat the encoder output seq_len times
        hidden = torch.unsqueeze(initial_hidden[-1], dim=0)
        h_shape = list(hidden.shape)
        h_shape[0] = seq_len
        hidden = hidden.expand(*h_shape)

        result = self.lstm(hidden)
        return result


class LSTMAEDecoderReverse(LSTMAEDecoder):
    """
    Reconstruct the time series in the opposite direction, starting with an initial hidden state from the encoder
    This corresponds to Malhotra 2016
    """
    def __init__(self, enc_hidden_dimension: int, hidden_dimensions: List[int], output_dimension: int):
        super().__init__(enc_hidden_dimension, hidden_dimensions, output_dimension)

        self.lstm = RNN(layer_type='LSTM', input_dimension=output_dimension, model='s2as',
                        hidden_dimensions=[enc_hidden_dimension] + hidden_dimensions)
        self.linear = torch.nn.Linear(hidden_dimensions[-1], output_dimension)

    def forward(self, initial_hidden: List[torch.Tensor], seq_len: int, x: torch.Tensor = None) -> torch.Tensor:
        """

        :param initial_hidden: tensor of shape (B, D)
        :param seq_len: int that determines the length of the produced sequence
        :param x: The ground truth sequence that should be reconstructed as a tensor of shape (T, B, D).
         This will be fed into the LSTM during training instead of the output from the previous step.
        :return: Tensor of shape (T, B, D)
        """
        # Produce the last output
        hidden = initial_hidden[::-1]
        output = [self.linear(hidden[-1].unsqueeze(0))]
        hidden = (hidden, [torch.zeros_like(h) for h in hidden])

        if self.training and x is not None:
            # Use actual time series input instead of predictions
            # Use only x_2, ..., x_T as inputs, since \hat{x}_1 is predicted from x_2
            # Inputs are reversed, since we want to generate \hat{x}_T, \hat{x}_{T-1}, ..., \hat{x}_1
            inputs = torch.flip(x[1:], dims=(0,))
            result = self.lstm(inputs, hidden_states=hidden)[-1]
            # Apply linear layer
            result = self.linear(result)
            result = torch.cat(output + [result], dim=0)
        else:
            # Generate sequence from scratch
            inputs = output

            for t in range(1, seq_len):
                input = inputs[t-1]
                out, hidden = self.lstm(input, hidden_states=hidden, return_hidden=True)
                output.append(self.linear(out[-1]))
                hidden = [(h.squeeze(0), c.squeeze(0)) for h, c in hidden]

            result = torch.cat(output, dim=0)

        # Reverse output to get the correct order
        result = torch.flip(result, dims=(0,))

        return result


def max_pool(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.max(x, dim=dim)[0]


class LSTMAE(AE, BaseModel):
    """
    Generic LSTMAE implementation
    """
    def __init__(self, input_dimension: int, hidden_dimensions=None, latent_pooling: Union[str, Callable] = 'last',
                 decoder_class: Type[LSTMAEDecoder] = LSTMAEDecoderReverse, return_latent: bool = False):
        if hidden_dimensions is None:
            hidden_dimensions = [40]

        encoder = RNN(layer_type='LSTM', input_dimension=input_dimension, model='s2as',
                      hidden_dimensions=hidden_dimensions)
        dec_hidden_dimensions = hidden_dimensions if len(hidden_dimensions) == 1 else hidden_dimensions[-2::-1]
        decoder = decoder_class(enc_hidden_dimension=hidden_dimensions[-1], hidden_dimensions=dec_hidden_dimensions,
                                output_dimension=input_dimension)

        super().__init__(encoder, decoder, return_latent=return_latent)

        if latent_pooling == 'last':
            self.latent_pooling = functools.partial(getitem, item=-1)
        elif latent_pooling == 'mean':
            self.latent_pooling = functools.partial(torch.mean, dim=0)
        elif latent_pooling == 'max':
            self.latent_pooling = functools.partial(max_pool, dim=0)
        elif callable(latent_pooling):
            self.latent_pooling = latent_pooling
        else:
            raise ValueError(f'Pooling method "{latent_pooling}" is not supported!')

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden = self.encoder(x)

        return [self.latent_pooling(h) for h in hidden]

    def forward(self, inputs: Tuple[torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param inputs: Tuple with a single tensor of shape (T, B, D)
        :return: tensor of shape (T, B, D)
        """
        x, = inputs
        seq_len = x.shape[0]
        hidden = self.encode(x)

        if self.training:
            pred = self.decoder(hidden, seq_len, x)
        else:
            pred = self.decoder(hidden, seq_len)

        if self.return_latent:
            return pred, hidden

        return pred


class LSTMAEMalhotra2016(LSTMAE):
    """
    Implementation of Malhotra 2016 (https://arxiv.org/pdf/1607.00148.pdf, default parameters)
    """
    def __init__(self, input_dimension: int, hidden_dimensions=None):
        super(LSTMAEMalhotra2016, self).__init__(input_dimension, hidden_dimensions=hidden_dimensions,
                                                 latent_pooling='last', decoder_class=LSTMAEDecoderReverse,
                                                 return_latent=False)


class LSTMAEMirza2018(LSTMAE):
    """
    Mirza 2018 (http://repository.bilkent.edu.tr/bitstream/handle/11693/50234/Computer_network_intrusion_detection_using_sequential_LSTM_neural_networks_autoencoders.pdf?sequence=1)
    """
    def __init__(self, input_dimension: int, hidden_dimensions: List[int] = [64], latent_pooling: str = 'mean'):
        super(LSTMAEMirza2018, self).__init__(input_dimension, hidden_dimensions=hidden_dimensions,
                                              latent_pooling=latent_pooling, decoder_class=LSTMAEDecoderSimple,
                                              return_latent=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs: Tuple[torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param inputs: Tuple with a single tensor of shape (T, B, D)
        :return: tensor of shape (T, B, D)
        """
        pred = super(LSTMAEMirza2018, self).forward(inputs)
        pred = self.sigmoid(pred)

        return pred


class LSTMAEAnomalyDetector(AnomalyDetector):
    def __init__(self, model: LSTMAE):
        super(LSTMAEAnomalyDetector, self).__init__()

        self.model = model

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

            target, = b_targets

            if i == 0:
                # Use all datapoints in the first window
                error = target - pred
            else:
                # In all subsequent windows, only the last datapoint will be new
                error = target[-1:] - pred[-1:]
            error.abs_()
            errors.append(error.reshape(-1, error.shape[-1]))

            mean += torch.sum(error, dim=(0, 1))
            total += error.shape[0] * error.shape[1]

        mean /= total

        errors = torch.cat(errors, dim=0)
        errors -= mean
        cov = torch.matmul(errors.T, errors)
        cov /= total - 1

        for i in range(5, 0, -1):
            try:
                cov.diagonal().add_(10**-i)
                cholesky = torch.linalg.cholesky(cov)
                if not torch.isnan(cholesky).any():
                    break
            except (torch.linalg.LinAlgError, RuntimeError):
                # If the covariance matrix is not positive definite, we can try to add a small value to the diagonal
                # until it becomes positive definite
                continue
        else:
            raise RuntimeError('Could not compute a valid covariance matrix!')

        precision = cov
        torch.cholesky_inverse(cholesky, out=precision)

        self.register_buffer('mean', mean)
        self.register_buffer('precision', precision)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B, D), output of shape (B,)
        with torch.no_grad():
            prediction = self.model(inputs)

        input, = inputs
        error = input[-1] - prediction[-1]
        error.abs_()

        error -= self.mean

        result = F.bilinear(error, error, self.precision.unsqueeze(0))
        return result.squeeze(-1)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[-1]
