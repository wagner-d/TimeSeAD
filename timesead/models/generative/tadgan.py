import itertools
from typing import Tuple, List, Iterator, Dict, Callable

import torch
from torch.nn import Parameter

from ..common import GAN, WassersteinGeneratorLoss, AnomalyDetector
from ...models import BaseModel
from ...optim.loss import Loss
from ...optim.trainer import Trainer
from ...utils import torch_utils
from ...utils.torch_utils import tensor2scalar


class TADGANEncoder(torch.nn.Module):
    def __init__(self, input_size: int, window_size: int, lstm_hidden_size: int = 100, latent_size: int = 20):
        super(TADGANEncoder, self).__init__()

        self.lstm = torch.nn.LSTM(input_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(2 * lstm_hidden_size * window_size, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, T, D)

        x, _ = self.lstm(x)
        # (B, T, 2*hidden_lstm)
        x = x.reshape(x.shape[0], -1)
        # (B, 2*T*hidden_lstm)
        x = self.linear(x)
        # (B, latent)

        return x


class TADGANGenerator(torch.nn.Module):
    def __init__(self, window_size: int, output_size: int, latent_size: int = 20, lstm_hidden_size: int = 64,
                 dropout: float = 0.2):
        super(TADGANGenerator, self).__init__()

        self.linear1 = torch.nn.Linear(latent_size, window_size // 2)
        self.lstm1 = torch.nn.LSTM(1, lstm_hidden_size, bidirectional=True, batch_first=True, dropout=dropout)
        self.upsample = torch.nn.Upsample(size=window_size, mode='nearest')
        self.lstm2 = torch.nn.LSTM(2 * lstm_hidden_size, lstm_hidden_size, bidirectional=True, batch_first=True,
                                   dropout=dropout)
        self.linear2 = torch.nn.Linear(2 * lstm_hidden_size, output_size)
        # We use Sigmoid as the final activation function, because we normalize data to [0, 1] instead of [-1, 1]
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, latent)

        x = self.linear1(x).unsqueeze(-1)
        # (B, T // 2, 1)
        x, _ = self.lstm1(x)
        # (B, T // 2, 2*hidden_lstm)
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
        # (B, T, 2*hidden_lstm)
        x, _ = self.lstm2(x)
        # (B, T, 2*hidden_lstm)
        # Apply linear layer at each time step independently
        x = self.linear2(x)
        # (B, T, out)
        x = self.final_activation(x)

        return x


class TADGANDiscriminatorX(torch.nn.Module):
    def __init__(self, input_size: int, window_size: int, conv_filters: int = 64, conv_kernel_size: int = 5,
                 dropout: float = 0.25):
        super(TADGANDiscriminatorX, self).__init__()

        self.conv1 = torch.nn.Conv1d(input_size, conv_filters, conv_kernel_size)
        self.conv2 = torch.nn.Conv1d(conv_filters, conv_filters, conv_kernel_size)
        self.conv3 = torch.nn.Conv1d(conv_filters, conv_filters, conv_kernel_size)
        self.conv4 = torch.nn.Conv1d(conv_filters, conv_filters, conv_kernel_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        out_length = window_size - 4 * (conv_kernel_size - 1)
        self.classification = torch.nn.Linear(out_length * conv_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, T, D)

        x = x.transpose(1, 2)  # Needed for conv layers
        # (B, D, T)
        x = self.dropout(self.leakyrelu(self.conv1(x)))
        # (B, conv_filters, T - (conv_kernel - 1))
        x = self.dropout(self.leakyrelu(self.conv2(x)))
        # (B, conv_filters, T - 2*(conv_kernel - 1))
        x = self.dropout(self.leakyrelu(self.conv3(x)))
        # (B, conv_filters, T - 3*(conv_kernel - 1))
        x = self.dropout(self.leakyrelu(self.conv4(x)))
        # (B, conv_filters, T - 4*(conv_kernel - 1))

        x = self.classification(x.view(x.shape[0], -1))
        # (B, 1)

        return x


class TADGANDiscriminatorZ(torch.nn.Module):
    def __init__(self, latent_size: int, hidden_size: int = 20, dropout: float = 0.2):
        super(TADGANDiscriminatorZ, self).__init__()

        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.classification = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, latent)

        x = self.dropout(self.leakyrelu(self.linear1(x)))
        # (B, hidden)
        x = self.dropout(self.leakyrelu(self.linear2(x)))
        # (B, hidden)

        x = self.classification(x.view(x.shape[0], -1))
        # (B, 1)

        return x


class TADGAN(BaseModel):
    def __init__(self, input_size: int, window_size: int, latent_size: int = 20, enc_lstm_hidden_size: int = 100,
                 gen_lstm_hidden_size: int = 64, disc_conv_filters: int = 64, disc_conv_kernel_size: int = 5,
                 disc_z_hidden_size: int = 20, gen_dropout: float = 0.2, disc_dropout: float = 0.25,
                 disc_z_dropout: float = 0.2):
        super(TADGAN, self).__init__()

        self.encoder = TADGANEncoder(input_size, window_size, enc_lstm_hidden_size, latent_size)
        self.generator = TADGANGenerator(window_size, input_size, latent_size, gen_lstm_hidden_size, gen_dropout)
        self.discriminatorx = TADGANDiscriminatorX(input_size, window_size, disc_conv_filters, disc_conv_kernel_size,
                                                   disc_dropout)
        self.discriminatorz = TADGANDiscriminatorZ(latent_size, disc_z_hidden_size, disc_z_dropout)

        self.gan = GAN(self.generator, self.discriminatorx)
        self.inverse_gan = GAN(self.encoder, self.discriminatorz)

        self.latent_size = latent_size

    def grouped_parameters(self) -> Tuple[Iterator[Parameter], ...]:
        return (self.discriminatorz.parameters(), self.discriminatorx.parameters(),
                itertools.chain(self.encoder.parameters(), self.generator.parameters()))

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        real_z, real_x = inputs
        # (B, latent), (B, T, D)

        fake_x, real_x_score, fake_x_score = self.gan((real_z, real_x))
        # (B, T, D), (B, 1), (B, 1)

        fake_z, real_z_score, fake_z_score = self.inverse_gan((real_x, real_z))
        # (B, latent), (B, 1), (B, 1)

        reconstructed_x = self.generator(fake_z)
        # (B, T, D)

        return fake_z, fake_x, real_z_score, fake_z_score, real_x_score, fake_x_score, reconstructed_x


class TADGANGeneratorLoss(WassersteinGeneratorLoss):
    def __init__(self, reconstruction_coeff: float = 10):
        super(TADGANGeneratorLoss, self).__init__()

        self.rec_coeff = reconstruction_coeff

        self.rec_loss = torch.nn.MSELoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_z, fake_x, real_z_score, fake_z_score, real_x_score, fake_x_score, reconstructed_x = predictions
        real_x, = targets

        gen_loss = super(TADGANGeneratorLoss, self).forward((fake_x, real_x_score, fake_x_score), targets)
        enc_loss = super(TADGANGeneratorLoss, self).forward((fake_z, real_z_score, fake_z_score), targets)
        rec_loss = self.rec_loss(fake_x, real_x)

        return gen_loss + enc_loss + self.rec_coeff * rec_loss


class TADGANTrainer(Trainer):
    def __init__(self, *args, disc_iterations: int = 5, **kwargs):
        super(TADGANTrainer, self).__init__(*args, **kwargs)

        self.disc_iterations = disc_iterations

        self._counter = 0
        self._last_generator_loss = 0

    def validate_batch(self, network: TADGAN, val_metrics: Dict[str, Callable],
                       b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> Dict[str, float]:
        real_x, = b_inputs
        real_target, = b_targets

        # Generate a random vector for z
        real_z = torch.randn(real_x.shape[0], network.latent_size, dtype=real_x.dtype, device=real_x.device)
        # (B, latent)

        res = network((real_z, real_x))
        fake_z, fake_x, real_z_score, fake_z_score, real_x_score, fake_x_score, reconstructed_x = res

        batch_metrics = {}
        for m_name, m in val_metrics.items():
            if m_name == 'loss_0':
                preds = fake_z, real_z_score, fake_z_score
                targets = real_z,
            elif m_name == 'loss_1':
                preds = fake_x, real_x_score, fake_x_score
                targets = b_targets
            else:
                preds = res
                targets = b_targets

            batch_metrics[m_name] = tensor2scalar(m(preds, targets, b_inputs, *args, **kwargs).detach().cpu()) \
                                    * b_inputs[0].shape[self.batch_dimension]

        return batch_metrics

    def train_batch(self, network: TADGAN, losses: List[Loss], optimizers: List[torch.optim.Optimizer], epoch: int,
                    num_epochs: int, b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...]) \
            -> List[float]:
        real_x, = b_inputs
        real_target, = b_targets

        disc_z_loss, disc_x_loss, gen_loss = losses
        disc_z_opt, disc_x_opt, gen_opt = optimizers

        # Generate a random vector for z
        real_z = torch.randn(real_x.shape[0], network.latent_size, dtype=real_x.dtype, device=real_x.device)
        # (B, latent)

        # Train Discriminator for z
        disc_z_loss, = super(TADGANTrainer, self).train_batch(network.inverse_gan, [disc_z_loss], [disc_z_opt], epoch,
                                                              num_epochs, (real_x, real_z), (real_z,))
        # Train Discriminator for x
        disc_x_loss, = super(TADGANTrainer, self).train_batch(network.gan, [disc_x_loss], [disc_x_opt], epoch,
                                                              num_epochs, (real_z, real_x), (real_target,))

        # Update generator and encoder only every few steps
        self._counter += 1
        if self._counter >= self.disc_iterations:
            self._last_generator_loss, = super(TADGANTrainer, self).train_batch(network, [gen_loss], [gen_opt],
                                                                                epoch, num_epochs,
                                                                                (real_z, real_x), b_targets)
            self._counter = 0

        return [disc_x_loss, disc_z_loss, self._last_generator_loss]


class TADGANAnomalyDetector(AnomalyDetector):
    def __init__(self, model: TADGAN, alpha: float = 0.5):
        super(TADGANAnomalyDetector, self).__init__()

        self.model = model
        self.alpha = alpha

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        rec_scores = []
        disc_scores = []
        total = 0

        rec_mean = 0
        disc_mean = 0
        # Compute mean over the entire validation dataset (minus the first few points)
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(torch_utils.get_device(self.model)) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(torch_utils.get_device(self.model)) for b_tar in b_targets)

            real_x, = b_inputs
            if real_x.ndim < 3:
                real_x = real_x.unsqueeze(0)

            with torch.no_grad():
                disc_score = self.model.discriminatorx(real_x).squeeze(-1)
                reconstructed_x = self.model.generator(self.model.encoder(real_x))

            target, = b_targets
            b_size = target.shape[1]

            error = real_x[:, -1] - reconstructed_x[:, -1]
            rec_score = torch_utils.batched_dot(error, error)

            rec_scores.append(rec_score)
            disc_scores.append(disc_score)

            rec_mean += torch.sum(rec_score)
            disc_mean += torch.sum(disc_score)
            total += b_size

        rec_mean /= total
        disc_mean /= total

        rec_scores = torch.cat(rec_scores, dim=0)
        disc_scores = torch.cat(disc_scores, dim=0)

        # Compute standard deviation
        rec_std = torch.sum((rec_scores - rec_mean) ** 2) / (total - 1)
        disc_std = torch.sum((disc_scores - disc_mean) ** 2) / (total - 1)

        self.register_buffer('rec_mean', rec_mean)
        self.register_buffer('disc_mean', disc_mean)
        self.register_buffer('rec_std', rec_std)
        self.register_buffer('disc_std', disc_std)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T, D), output of shape (B,)
        real_x, = inputs

        with torch.no_grad():
            disc_score = self.model.discriminatorx(real_x).squeeze(-1)
        disc_score -= self.disc_mean
        disc_score /= self.disc_std
        torch.abs(disc_score, out=disc_score)

        with torch.no_grad():
            reconstructed_x = self.model.generator(self.model.encoder(real_x))
        error = real_x[:, -1] - reconstructed_x[:, -1]
        rec_score = torch.sum(error ** 2, dim=-1)
        rec_score -= self.rec_mean
        rec_score /= self.rec_std
        torch.abs(rec_score, out=rec_score)

        return self.alpha * rec_score + (1 - self.alpha) * disc_score

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[:, -1]