from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


def calc_causal_same_pad(kernel_size: int, stride: int = 1, in_shape: int = 1, dilation: int = 1) -> int:
    return in_shape * (stride - 1) - stride + dilation * (kernel_size - 1) + 1


def calc_same_pad(kernel_size: int, stride: int = 1, in_shape: int = 1, dilation: int = 1) -> Tuple[int, int]:
    total_pad = calc_causal_same_pad(kernel_size, stride, in_shape, dilation)
    pad_start = total_pad // 2
    pad_end = total_pad - pad_start

    return pad_start, pad_end


class SameZeroPad1d(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, in_shape: int = 1, dilation: int = 1):
        """
        Replicates the "SAME" pad algorithm from Tensorflow. Note that Tensorflow will always assume stride = 1,
        whereas this implementation also takes different strides into account.

        :param kernel_size: Kernel size that will be used
        :param stride: Stride that will be used
        :param in_shape: Size of the input. This is only needed if stride != 1
        :param dilation: Dilation that will be used
        """
        super(SameZeroPad1d, self).__init__()

        self.padding = calc_same_pad(kernel_size, stride, in_shape, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, mode='constant', value=0)


class SameCausalZeroPad1d(SameZeroPad1d):
    def __init__(self, kernel_size: int, stride: int = 1, in_shape: int = 1, dilation: int = 1):
        """
        Replicates the "causal" pad algorithm from Tensorflow. Note that Tensorflow will always assume stride = 1,
        whereas this implementation also takes different strides into account.

        :param kernel_size: Kernel size that will be used
        :param stride: Stride that will be used
        :param in_shape: Size of the input. This is only needed if stride != 1
        :param dilation: Dilation that will be used
        """
        super(SameCausalZeroPad1d, self).__init__(kernel_size, stride, in_shape, dilation)

        # Causal padding means that zeros are only added before the start of the sequence
        self.padding = (self.padding[0] + self.padding[1], 0)


class SameZeroPad2d(torch.nn.ZeroPad2d):
    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t = 1, in_shape: _size_2_t = 1, dilation: _size_2_t = 1):
        """
        Replicates the "SAME" pad algorithm from Tensorflow. Note that Tensorflow will always assume stride = 1,
        whereas this implementation also takes different strides into account.

        :param kernel_size: Kernel size that will be used
        :param stride: Stride that will be used
        :param in_shape: Size of the input. This is only needed if stride != 1
        :param dilation: Dilation that will be used
        """
        from torch.nn.modules.utils import _pair
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        in_shape = _pair(in_shape)
        dilation = _pair(dilation)

        # Horizontal
        pad_left, pad_right = calc_same_pad(kernel_size[1], stride[1], in_shape[1], dilation[1])

        # Vertical
        pad_top, pad_bottom = calc_same_pad(kernel_size[0], stride[0], in_shape[0], dilation[0])

        super(SameZeroPad2d, self).__init__((pad_left, pad_right, pad_top, pad_bottom))
