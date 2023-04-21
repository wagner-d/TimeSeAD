import itertools
from inspect import Parameter
from typing import List, Tuple, Iterator, Dict, Callable

import torch

from .lstm_vae import RNNVAEGaussianEncoder
from ..common import RNN, VAE, AnomalyDetector
from ...models import BaseModel
from ...optim.loss import Loss
from ...optim.trainer import Trainer
from ...utils.torch_utils import tensor2scalar


class LSTMVAEGANDecoder(torch.nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60], latent_dim: int = 10):
        super(LSTMVAEGANDecoder, self).__init__()

        self.rnn = RNN('lstm', 's2s', latent_dim, lstm_hidden_dims)
        self.linear_mean = torch.nn.Linear(lstm_hidden_dims[-1], input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        rnn_out = self.rnn(x)

        mean = self.linear_mean(rnn_out)

        # The paper always assumes std of one
        return mean


class LSTMVAEGANDiscriminator(torch.nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60]):
        super(LSTMVAEGANDiscriminator, self).__init__()

        self.rnn = RNN('lstm', 's2s', input_dim, lstm_hidden_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        rnn_out = self.rnn(x)

        return rnn_out, torch.ones_like(rnn_out)


class LSTMVAEGAN(BaseModel):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60], latent_dim: int = 10):
        super(LSTMVAEGAN, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = RNNVAEGaussianEncoder(input_dim, rnn_type='lstm', rnn_hidden_dims=lstm_hidden_dims,
                                             latent_dim=latent_dim)
        self.decoder = LSTMVAEGANDecoder(input_dim, lstm_hidden_dims, latent_dim)
        self.discriminator = LSTMVAEGANDiscriminator(input_dim, lstm_hidden_dims)
        self.classifier = torch.nn.Linear(lstm_hidden_dims[-1], 1)

        self.vae = VAE(self.encoder, torch.nn.Sequential(self.decoder, self.discriminator))

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # z: random sample (T, B, latent_dim)
        # x: (T, B, D)
        z, x = inputs

        z_mu, z_log_var, x_rec, _ = self.vae(x)
        x, _ = self.discriminator(x)
        x_gen, _ = self.discriminator(self.decoder(z))

        x_score = self.classifier(x)
        x_rec_score = self.classifier(x_rec)
        x_gen_score = self.classifier(x_gen)

        return z_mu, z_log_var, x, x_rec, x_score, x_rec_score, x_gen_score

    def grouped_parameters(self) -> Tuple[Iterator[Parameter], ...]:
        return self.encoder.parameters(), self.decoder.parameters(), \
               itertools.chain(self.discriminator.parameters(), self.classifier.parameters())


class LSTMVAEGANTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(LSTMVAEGANTrainer, self).__init__(*args, **kwargs)

    def validate_batch(self, network: LSTMVAEGAN, val_metrics: Dict[str, Callable],
                       b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> Dict[str, float]:
        real_x, = b_inputs
        real_target, = b_targets

        # Generate a random vector for z
        real_z = torch.randn(real_x.shape[0], real_x.shape[1], network.latent_dim, dtype=real_x.dtype, device=real_x.device)
        # (T, B, latent)

        res = network((real_z, real_x))
        z_mu, z_log_var, x, x_rec, x_score, x_rec_score, x_gen_score = res

        batch_metrics = {}
        for m_name, m in val_metrics.items():
            if m_name == 'loss_0':  # VAE loss
                preds = z_mu, z_log_var, x_rec, torch.ones_like(x_rec)
                targets = x,
            elif m_name == 'loss_1':  # Generator loss
                preds = None, None, x_rec_score
                targets = None,
            elif m_name == 'loss_2':  # Discriminator loss
                preds = x_rec, x_score, torch.cat([x_rec_score, x_gen_score], dim=1)
                targets = None,
            else:  # Discriminator loss
                preds = res
                targets = b_targets

            batch_metrics[m_name] = tensor2scalar(m(preds, targets, b_inputs, *args, **kwargs).detach().cpu()) \
                                       * b_inputs[0].shape[self.batch_dimension]

        return batch_metrics

    def train_batch(self, network: LSTMVAEGAN, losses: List[Loss], optimizers: List[torch.optim.Optimizer], epoch: int,
                    num_epochs: int, b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...]) \
            -> List[float]:
        real_x, = b_inputs
        real_target, = b_targets

        vae_loss, gen_loss, disc_loss = losses
        enc_opt, dec_opt, disc_opt = optimizers

        x_target, _ = network.discriminator(real_target)

        # Train encoder
        z_mu, z_log_var, x_rec, x_rec_std = network.vae(real_x)
        l_vae = vae_loss((z_mu, z_log_var, x_rec, x_rec_std), (x_target,))

        enc_opt.zero_grad(True)
        l_vae.backward(inputs=list(network.encoder.parameters()))
        enc_opt.step()
        enc_opt.zero_grad(True)
        enc_loss = tensor2scalar(l_vae.detach().cpu()) * b_inputs[0].shape[self.batch_dimension]

        # Train decoder
        z_mu, z_log_var, x_rec, x_rec_std = network.vae(real_x)
        l_vae = vae_loss((z_mu, z_log_var, x_rec, x_rec_std), (x_target,))

        x_rec_score = network.classifier(x_rec)
        l_disc = gen_loss((None, None, x_rec_score), (None,))

        l = l_vae + l_disc

        dec_opt.zero_grad(True)
        l.backward(inputs=list(network.decoder.parameters()))
        dec_opt.step()
        dec_opt.zero_grad(True)
        dec_loss = tensor2scalar(l.detach().cpu()) * b_inputs[0].shape[self.batch_dimension]

        # Train discriminator
        # Generate a random vector for z
        real_z = torch.randn(real_x.shape[0], real_x.shape[1], network.latent_dim, dtype=real_x.dtype,
                             device=real_x.device)
        # (T, B, latent)

        z_mu, z_log_var, x, x_rec, x_score, x_rec_score, x_gen_score = network((real_z, real_x))

        l_disc = disc_loss((x_rec, x_score, x_rec_score), (None,))

        disc_opt.zero_grad(True)
        l_disc.backward(inputs=list(itertools.chain(network.discriminator.parameters(), network.classifier.parameters())))
        disc_opt.step()
        disc_opt.zero_grad(True)
        disc_loss = tensor2scalar(l_disc.detach().cpu()) * b_inputs[0].shape[self.batch_dimension]

        return [enc_loss, dec_loss, disc_loss]


class LSTMVAEGANAnomalyDetector(AnomalyDetector):
    def __init__(self, model: LSTMVAEGAN, alpha: float = 0.5):
        super(LSTMVAEGANAnomalyDetector, self).__init__()

        self.model = model
        self.alpha = alpha

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B, D), output of shape (B,)
        real_x, = inputs

        with torch.no_grad():
            disc_score, _ = self.model.discriminator(real_x)
            disc_score = self.model.classifier(disc_score[-1]).squeeze(-1)
            z_mean, z_std = self.model.encoder(real_x)
            x_rec = self.model.decoder(z_mean)

        error = torch.abs(real_x[-1] - x_rec[-1])
        rec_score = torch.sum(error, dim=-1)

        return (1 - self.alpha) * rec_score - self.alpha * disc_score

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[-1]
