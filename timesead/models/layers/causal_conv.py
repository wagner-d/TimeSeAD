import torch
from torch.nn.common_types import _size_1_t

from .same_pad import calc_causal_same_pad


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ):
        self.causal_padding = calc_causal_same_pad(kernel_size, stride=stride, dilation=dilation)
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                           padding=self.causal_padding, dilation=dilation, groups=groups, bias=bias,
                                           padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, D, T)
        result = super(CausalConv1d, self).forward(input)

        return result[..., :-self.causal_padding]
