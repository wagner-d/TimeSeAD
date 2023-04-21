import math
from typing import Tuple, Union
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hid_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            spatial_size: Tuple[int, int]):
        super(ConvLSTMCell, self).__init__()

        self.hid_channels = hid_channels

        self.x2h = nn.Conv2d(in_channels, 4 * hid_channels, kernel_size, bias=True, padding='same')
        self.h2h = nn.Conv2d(hid_channels, 4 * hid_channels, kernel_size, bias=False, padding='same')
        self.c2c = nn.Parameter(
            torch.empty(
                1,
                3 * hid_channels,
                *spatial_size,
                dtype=self.x2h.weight.dtype,
            ))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hid_channels)
        torch.nn.init.uniform_(self.c2c, -stdv, stdv)

    def forward(self, x, h, c):
        # Perform compact convolution
        x_ifo, xc = self.x2h(x).split(3 * self.hid_channels, dim=1)
        h_ifo, hc = self.h2h(h).split(3 * self.hid_channels, dim=1)
        # Perform Hadamard product
        c_ifo = self.c2c * c.repeat_interleave(3, dim=1)

        i, f, o = torch.sigmoid(x_ifo + h_ifo + c_ifo).split(self.hid_channels, dim=1)

        memory = f * c + i * torch.tanh(xc + hc)
        hidden = o * torch.tanh(memory)

        return hidden, memory


class ConvLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvLSTM, self).__init__()
        self.lstm = ConvLSTMCell(*args, **kwargs)

    def forward(self, x, hidden=None, memory=None):
        """
        input shape: (T, B, C, H, W)
        """
        if hidden is None:
            hidden = torch.zeros(
                x.size(1),
                self.lstm.hid_channels,
                x.size(3),
                x.size(4),
                device=self.lstm.x2h.weight.device,
            )
        if memory is None:
            memory = torch.zeros(
                x.size(1),
                self.lstm.hid_channels,
                x.size(3),
                x.size(4),
                device=self.lstm.x2h.weight.device,
            )

        output = []
        for data in x:
            hidden, memory = self.lstm(data, hidden, memory)
            output.append(hidden.unsqueeze(0))

        return torch.cat(output), (hidden, memory)
