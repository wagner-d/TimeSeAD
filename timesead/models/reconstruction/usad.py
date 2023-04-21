# Partly taken from https://github.com/manigalati/usad/blob/master/usad.py
#
# BSD License
#
# Copyright (c) 2020, EURECOM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
import itertools
from typing import Tuple

import torch

from ..common import MLP, AnomalyDetector
from ...models import BaseModel
from ...optim.loss import Loss


class BasicAE(BaseModel):
    """
    What I believe to be the basic "AE" model from the USAD paper
    """
    def __init__(self, w_size: int, z_size: int = 40):
        super().__init__()
        self.encoder = MLP(w_size, [w_size // 2, w_size // 4], z_size, activation=torch.nn.ReLU(True),
                           activation_after_last_layer=True)
        self.decoder = torch.nn.Sequential(
            MLP(z_size, [w_size // 4, w_size // 2], w_size, activation=torch.nn.ReLU(True)),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        x, = inputs
        x_shape = x.shape
        x = x.view(x_shape[0], -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat.view(x_shape)


class USADModel(BaseModel):
    def __init__(self, w_size: int, z_size: int):
        super().__init__()
        self.encoder = MLP(w_size, [w_size // 2, w_size // 4], z_size, activation=torch.nn.ReLU(),
                           activation_after_last_layer=True)
        self.decoder1 = torch.nn.Sequential(
            MLP(z_size, [w_size // 4, w_size // 2], w_size, activation=torch.nn.ReLU()),
            torch.nn.Sigmoid()
        )
        self.decoder2 = torch.nn.Sequential(
            MLP(z_size, [w_size // 4, w_size // 2], w_size, activation=torch.nn.ReLU()),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """

        :param inputs: Tuple with one tensor of shape (B, T, D)
        :return:
        """
        x, = inputs
        x_shape = x.shape
        # Flatten time dimension
        x = x.view(x_shape[0], -1)

        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        # Restore time dimension
        return w1.view(x_shape), w2.view(x_shape), w3.view(x_shape)

    def grouped_parameters(self):
        return itertools.chain(self.encoder.parameters(), self.decoder1.parameters()), \
               itertools.chain(self.encoder.parameters(), self.decoder2.parameters())


class USADDecoder1Loss(Loss):
    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args,
                **kwargs) -> torch.Tensor:
        w1, w2, w3 = predictions
        x, = targets
        n = kwargs.get('epoch', 0) + 1

        return 1 / n * torch.mean((x - w1) ** 2) + (1 - 1 / n) * torch.mean((x - w3) ** 2)


class USADDecoder2Loss(Loss):
    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args,
                **kwargs) -> torch.Tensor:
        w1, w2, w3 = predictions
        x, = targets
        n = kwargs.get('epoch', 0) + 1

        return 1 / n * torch.mean((x - w2) ** 2) - (1 - 1 / n) * torch.mean((x - w3) ** 2)


class USADAnomalyDetector(AnomalyDetector):
    def __init__(self, model: USADModel, alpha: float = 0.5):
        super(USADAnomalyDetector, self).__init__()

        self.model = model
        self.alpha = alpha
        self.beta = 1 - alpha

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T, D), output of shape (B,)
        with torch.no_grad():
            prediction = self.model(inputs)

        w1, w2, w3 = prediction
        x, = inputs
        return self.alpha * torch.mean((x[:, -1] - w1[:, -1]) ** 2, dim=-1) + \
               self.beta * torch.mean((x[:, -1] - w2[:, -1]) ** 2, dim=-1)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target, = targets
        # Just return the last label of the window
        return target[:, -1]
