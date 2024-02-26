# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from ...models import BaseModel
from ..layers.embed import DataEmbedding
from ..layers.inception import InceptionBlockV1


def FFT_for_Period(x: torch.tensor, k: int=2) -> Tuple[torch.tensor, torch.tensor]:
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
            self,
            window_size: int,
            top_k: int=5,
            d_model: int=64,
            d_ff: int=64,
            num_kernels: int=8
        ) -> None:
        super(TimesBlock, self).__init__()
        # TODO(AR): check if window_size is needed
        self.seq_len = window_size
        self.top_k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # TODO(AR): check if the padding is necessary
            # padding
            if self.seq_len % period != 0:
                length = ( (self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(BaseModel):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        top_k: int=5,
        d_model: int=64,
        d_ff: int=64,
        num_kernels: int=8,
        e_layers: int=2,
        dropout: float=0.1
    ) -> None:
        super(TimesNet, self).__init__()
        # Rename to window_size
        self.seq_len = window_size
        self.model = nn.ModuleList([TimesBlock(window_size, top_k, d_model, d_ff, num_kernels)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)

        self.projection = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x_enc = inputs[0]
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        return dec_out

