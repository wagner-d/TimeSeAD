"""
Abstract class implementing the general interface of a loss.
"""
import abc
import math
from typing import Tuple, Union

import torch
from torch.nn import functional as F


class Loss(torch.nn.modules.loss._Loss, abc.ABC):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(Loss, self).__init__(size_average, reduce, reduction)

    @abc.abstractmethod
    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        raise NotImplementedError()


class TorchLossWrapper(Loss):
    def __init__(self, torch_loss: torch.nn.modules.loss._Loss, size_average=None, reduce=None,
                 reduction: str = 'mean'):
        super(Loss, self).__init__(size_average, reduce, reduction)

        self.torch_loss = torch_loss

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        assert len(predictions) == len(targets)

        loss = 0
        for pred, target in zip(predictions, targets):
            loss += self.torch_loss(pred, target)

        return loss


class LogCoshLoss(Loss):
    def logcosh(self, x: torch.Tensor) -> torch.Tensor:
        # log( (exp(x) + exp(-x) ) / 2)
        result = F.softplus(-2 * x) + x - math.log(2)

        if self.reduction == 'none':
            return result

        if self.reduction == 'sum':
            return torch.sum(result)

        # Default: mean
        return torch.mean(result)

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> \
    Union[torch.Tensor, Tuple[torch.Tensor]]:
        return sum(self.logcosh(t - p) for t, p in zip(targets, predictions))