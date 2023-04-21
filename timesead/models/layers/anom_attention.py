# Models for Anomaly Transformer
# This code is taken from https://github.com/thuml/Anomaly-Transformer/blob/main/model/attn.py

import math
from typing import Optional
import numpy as np

import torch
import torch.nn as nn


class TriangularCausalMask:
    def __init__(self, B, L, device: str = "cpu") -> None:
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(
        self,
        win_size: int,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        output_attention: bool = False,
    ) -> None:
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.register_buffer(
            "distances", torch.zeros(1, 1, win_size, win_size), persistent=False
        )
        for i in range(win_size):
            for j in range(win_size):
                self.distances[:, :, i, j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        window_size = attn.shape[-1]
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L

        prior = self.distances.repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = (
            1.0
            / (math.sqrt(2 * math.pi) * sigma)
            * torch.exp(-(prior**2) / 2 / (sigma**2))
        )
        # Given the paper, I think this should be:
        # prior = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * (
        #     torch.exp(-prior.pow(2) / (2 * sigma.pow(2)))
        # )
        # Moreover, better to to prior rescale here then later in final Transformer Model.
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ) -> None:
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        # Given the paper, I think the Linears should be without biases:
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries, keys, values, sigma, attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
