# Code from official implementation: https://github.com/thuml/Anomaly-Transformer

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import AnomalyDetector
from ..layers import AnomalyAttention, AttentionLayer, DataEmbedding
from ...models import BaseModel
from ...optim.loss import Loss
from ...optim.trainer import Trainer
from ...utils.torch_utils import tensor2scalar


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(
        self, attn_layers: List[nn.Module], norm_layer: Optional[nn.Module] = None
    ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask=None):
        # in shape: [bz, win, d_model]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(BaseModel):
    def __init__(
        self,
        win_size: int,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_attention: bool = True,
    ) -> None:
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(input_dim, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size,
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # x in shape [bz, win, feats]
        enc_out = self.embedding(x)  # out shape: [bz, win, d_model]
        enc_out, series, prior, sigmas = self.encoder(enc_out)

        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return (enc_out,)  # out shape [batch_size, seq_len, num_features]

def symm_kl_loss(p, q, eps=1e-4, reduce=True):
    res1 = torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)
    res2 = torch.sum(q * (torch.log(q + eps) - torch.log(p + eps)), dim=-1)
    if reduce:
        return res1.mean() + res2.mean()
    else:
        return res1.mean(dim=1) + res2.mean(dim=1)

class AnomTransf_Loss(Loss):
    def __init__(self, lamb: float = 3.0):
        super(AnomTransf_Loss, self).__init__()

        self.lamb = lamb
        self.mse_loss = nn.MSELoss()

    def calc_association_disc(
        self, series: List[torch.Tensor], priors: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        series_loss = 0.0
        prior_loss = 0.0
        for serie, prior in zip(series, priors):
            prior_rescale = prior.div(torch.sum(prior, dim=-1, keepdim=True))
            # Calc series loss
            series_loss += symm_kl_loss(serie, prior_rescale.detach())
            # Calc prior loss
            prior_loss += symm_kl_loss(prior_rescale, serie.detach())

        series_loss = series_loss / len(priors)
        prior_loss = prior_loss / len(priors)

        return series_loss, prior_loss

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        output, series, prior, _ = predictions
        (input,) = targets

        # Calc losses
        series_loss, prior_loss = self.calc_association_disc(series, prior)
        recon_loss = self.mse_loss(output, input)

        loss1 = recon_loss - self.lamb * series_loss
        loss2 = recon_loss + self.lamb * prior_loss

        return loss1, loss2, recon_loss


class AnomTransf_Trainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(AnomTransf_Trainer, self).__init__(*args, **kwargs)

    def validate_batch(self, network: torch.nn.Module, val_metrics: Dict[str, Callable],
                       b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> Dict[str, float]:
        (x,) = b_inputs

        batch_metrics = {}
        for m_name, m in val_metrics.items():
            _, loss2, recon_loss = m(network(x), b_targets, *args, **kwargs)
            batch_metrics['loss_0'] = tensor2scalar(recon_loss.detach().cpu()) * b_inputs[0].shape[self.batch_dimension]
            batch_metrics['loss_1'] = tensor2scalar((loss2-recon_loss).detach().cpu()) * b_inputs[0].shape[self.batch_dimension]

        return batch_metrics

    def train_batch(
        self,
        network: AnomalyTransformer,
        losses: List[Loss],
        optimizers: List[torch.optim.Optimizer],
        epoch: int,
        num_epochs: int,
        b_inputs: Tuple[torch.Tensor, ...],
        b_targets: Tuple[torch.Tensor, ...],
    ) -> List[float]:
        (x,) = b_inputs

        opt, *_ = optimizers
        loss, *_ = losses

        # Forward pass
        loss1, loss2, recon_loss = loss(network(x), b_targets)

        # Backward pass - Minimax strategy
        opt.zero_grad(True)
        loss1.backward(retain_graph=True)
        loss2.backward()
        opt.step()

        return [loss1.item(), loss2.item(), recon_loss.item()]


class AnomTransf_AnomalyDetector(AnomalyDetector):
    def __init__(self, model: AnomalyTransformer) -> None:
        super(AnomTransf_AnomalyDetector, self).__init__()

        self.model = model

    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        # Input of shape [B, T, F] output of shape (B,)
        (x,) = inputs

        # Temperatur from online implementation
        temperature = 50.

        with torch.no_grad():
            output, series, priors, _ = self.model(x)
            # Calc recon loss with mean over features: out shape [B, T]
            recon_loss = F.mse_loss(x, output, reduction='none').mean(dim=-1)

            # Init losses
            series_loss = 0.0
            prior_loss = 0.0
            for serie, prior in zip(series,priors):
                prior_norm = prior.div(torch.sum(prior, dim=-1, keepdim=True))
                series_loss += symm_kl_loss(serie, prior_norm, reduce=False) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            anomaly_scores = metric * recon_loss

        # Return last observations
        return anomaly_scores[:,-1]

    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T) output of shape (B)
        (target,) = targets
        # Just return the last label of the window
        return target[:, -1]

