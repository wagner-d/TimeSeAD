# Part of this is taken from https://github.com/wang-chen/kervolution/blob/unfold/kervolution.py
# Kervolution Neural Networks
# This file is part of the project of Kervolutional Neural Networks
# It implements the kervolution for 4D tensors in Pytorch
# Copyright (C) 2018 [Wang, Chen] <wang.chen@zoho.com>
# Nanyang Technological University (NTU), Singapore.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import abc
from typing import Union

import torch

from ...utils import torch_utils


class Kernel(torch.nn.Module, abc.ABC):
    def __init__(self, learnable_parameters: bool = False):
        super().__init__()

        self.learnable_parameters = learnable_parameters

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag_only: bool = False) -> torch.Tensor:
        """
        Compute the kernel function for inputs x1 and x2

        :param x1: A tensor of shape ([B1], D)
        :param x2: A tensor of shape ([B2], D)
        :param diag_only: Whether the entire kernel matrix should be computed or only the diagonal
        :return: A tensor of shape ([B1] + [B2]) if diag_only = False, else a tensor of shape ([B]), where [B] is the
            result of broadcasting [B1] and [B2].
        """
        if diag_only:
            return self._forward_diag(x1, x2)

        # Reshape inputs
        x1shape = x1.shape
        x2shape = x2.shape
        x1 = x1.view(-1, x1shape[-1])
        x2 = x2.view(-1, x2shape[-1])

        result = self._forward_full(x1, x2)

        return result.view(*(x1shape[:-1] + x2shape[:-1]))

    @abc.abstractmethod
    def _forward_full(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute the full kernel matrix for inputs x1 and x2.

        :param x1: A tensor of shape (B1, D)
        :param x2: A tensor of shape (B2, D)
        :return: A tensor K of shape (B1, B2), where K[i, j] = k(x1[i], x2[j])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _forward_diag(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute the diagonal of the kernel matrix for inputs x1 and x2.

        :param x1: A tensor of shape ([B1], D)
        :param x2: A tensor of shape ([B2], D)
        :return: A tensor K of shape ([B]), where K[i,...] = k(x1[i,...], x2[i,...]) and [B] is the result of
            broadcasting dimensions [B1] and [B2]
        """
        raise NotImplementedError


class LinearKernel(Kernel):
    def _forward_full(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.matmul(x1, x2.T)

    def _forward_diag(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch_utils.batched_dot(x1, x2)


class PolynomialKernel(Kernel):
    def __init__(self, degree: int = 2, c0: float = 0.0, learnable_parameters: bool = False):
        super(PolynomialKernel, self).__init__(learnable_parameters)

        self.degree = degree

        self.c0 = c0
        if self.learnable_parameters:
            self.c0 = torch.nn.Parameter(torch.tensor(c0, dtype=torch.float))

    def _forward_full(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.pow(torch.matmul(x1, x2.T) + self.c0, self.degree)

    def _forward_diag(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.pow(torch_utils.batched_dot(x1, x2) + self.c0, self.degree)


class RBFKernel(Kernel):
    def __init__(self, gamma: float = 1.0, learnable_parameters: bool = False):
        super(RBFKernel, self).__init__(learnable_parameters)

        self.gamma = gamma
        if self.learnable_parameters:
            self.gamma = torch.nn.Parameter(torch.tensor(gamma, dtype=torch.float))

    def _forward_full(self, x1: torch.Tensor, x2: torch.Tensor):
        sqdist = torch.sum(x1**2, dim=-1, keepdim=True) \
                 - 2*torch.matmul(x1, x2.T) \
                 + torch.sum(x2**2, dim=-1, keepdim=True).T

        return torch.exp(-self.gamma * sqdist)

    def _forward_diag(self, x1: torch.Tensor, x2: torch.Tensor):
        sqdist = torch.sum((x1 - x2)**2, dim=-1)
        return torch.exp(-self.gamma * sqdist)


kernels = {
    'linear': LinearKernel,
    'polynomial': PolynomialKernel,
    'rbf': RBFKernel
}


class Kerv1d(torch.nn.Conv1d):
    r"""
    Applies a 1D kervolution over an input signal composed of several inputplanes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        kernel (str or Kernel), Default: 'linear'
        learnable_kernel (bool): Learnable kernel parameters.  Default: False
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

           .. math::

              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Examples:
        >>> m = Kerv1d(16, 33, 3, kernel='rbf')
        >>> input = torch.randn(20, 16, 70)
        >>> output = m(input)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', kernel: Union[str, Kernel] = 'linear',
                 learnable_kernel: bool = False):

        super(Kerv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                     padding_mode)
        self.unfold = torch.nn.Unfold((kernel_size, 1), (dilation, 1), (padding, 0), (stride, 1))

        if isinstance(kernel, str):
            kernel = kernels[kernel](learnable_parameters=learnable_kernel)
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input and weight to match a common shape of (B, out_channels, n_patches, D) after broadcasting
        x = self.unfold(x.unsqueeze(-1)).unsqueeze(1).transpose(-1, -2)
        weight = self.weight.view(1, self.out_channels, 1, -1)

        # (B, 1, n_patches, D), (1, out_channels, 1, D) -> (B, out_channels, n_patches)
        output = self.kernel(x, weight, diag_only=True)

        if self.bias is not None:
            output = output + self.bias.view(1, self.out_channels, 1)

        return output
