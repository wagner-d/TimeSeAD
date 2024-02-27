# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Tuple

import torch
import torch.nn as nn

from ...models import BaseModel
from ..layers import DataEmbedding
from ..layers import AutoCorrelation, AutoCorrelationLayer
from ..layers.autoformer_encdec import Encoder, EncoderLayer, CustomLayerNorm, SeriesDecomp


class Autoformer(BaseModel):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(
            self,
            window_size: int,
            input_dim: int,
            moving_avg: int=25,
            model_dim: int=128,
            dropout: float=0.1,
            attention_factor: int=1,
            num_heads: int=8,
            fcn_dim: int=128,
            activation: str='gelu',
            encoder_layers: int=3,
        ) -> None:
        super(Autoformer, self).__init__()
        self.seq_len = window_size

        # Decomp
        kernel_size = moving_avg
        self.decomp = SeriesDecomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding(input_dim, model_dim, dropout, use_pos=False)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(attention_factor, attention_dropout=dropout,
                                        output_attention=False),
                        model_dim, num_heads),
                    model_dim,
                    fcn_dim,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(encoder_layers)
            ],
            norm_layer=CustomLayerNorm(model_dim)
        )
        # Decoder
        self.projection = nn.Linear(model_dim, input_dim, bias=True)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x_enc = inputs[0]
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

