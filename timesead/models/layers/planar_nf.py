# Taken from https://github.com/e-hulten/planar-flows/
from typing import Tuple

import torch
import torch.nn.functional as F


class PlanarTransform(torch.nn.Module):
    """
    Implementation of the invertible transformation used in planar flow
        f(z) = z + u * h(dot(w.T, z) + b)

    See Section 4.1 in https://arxiv.org/pdf/1505.05770.pdf.
    """

    def __init__(self, dim: int, epsilon: float = 1e-4):
        """Initialise weights and bias.

        :param dim: Dimensionality of the distribution to be estimated.
        """
        super().__init__()

        self.epsilon = epsilon

        w = torch.randn(dim)
        w *= 0.1
        self.w = torch.nn.Parameter(w)

        b = torch.randn(1)
        b *= 0.1
        self.b = torch.nn.Parameter(b)

        u = torch.randn(dim)
        u *= 0.1
        self.u = torch.nn.Parameter(u)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.get_u_hat()

        a = torch.tanh(torch.inner(z, self.w) + self.b).unsqueeze(-1)
        y = z + self.u * a

        psi = (1 - a**2) * self.w
        abs_det = (1 + torch.inner(self.u, psi)).abs()
        log_det = torch.log(self.epsilon + abs_det)

        return y, log_det

    def get_u_hat(self) -> None:
        """
        Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition
        for invertibility of the transformation f(z). See Appendix A.1.
        """
        wtu = torch.inner(self.u, self.w)
        if wtu >= -1:
            return

        m_wtu = -1 + F.softplus(wtu)
        self.u.data = self.u + (m_wtu - wtu) * self.w / torch.dot(self.w, self.w)


class PlanarFlow(torch.nn.Module):
    def __init__(self, dim: int, num_layers: int = 6):
        """
        Make a planar flow by stacking planar transformations in sequence.

        :param dim: Dimensionality of the distribution to be estimated.
        :param num_layers: Number of transformations in the flow.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([PlanarTransform(dim) for _ in range(num_layers)])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_sum = 0

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum = log_det + log_det_sum

        return z, log_det_sum
