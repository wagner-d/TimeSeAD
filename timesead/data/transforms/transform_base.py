import abc
from typing import Tuple, Optional, List, Union

import torch


class Transform(abc.ABC):
    """
    Base class for all transforms.
    A Transform processes one (or several) data points and outputs them. Transforms can be chained in a pull-based
    pipeline.
    """
    def __init__(self, parent: Optional['Transform']):
        """
        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
            Can be `None` in the case of a source.
        """
        self.parent = parent

    def get_datapoint(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Returns a datapoint (in our case this is a sequence) from this transform.

        :param item: Must be `0<=item<len(self)`
        :return: A datapoint of the form `(inputs, targets)`, where `inputs` and `targets` are tuples of tensors.
        """
        if not (0 <= item < len(self)):
            raise IndexError

        return self._get_datapoint_impl(item)

    @abc.abstractmethod
    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        This should be implemented by every subclass to produce a datapoint.

        :param item: Index of the datapoint to fetch
        :return: A datapoint of the form `(inputs, targets)`, where `inputs` and `targets` are tuples of tensors.
        """
        raise NotImplementedError

    def __len__(self) -> Optional[int]:
        """
        This should return the number of available sequences after the transformation.
        """
        return len(self.parent) if self.parent is not None else None

    @property
    def seq_len(self) -> Union[int, List[int]]:
        """
        This should return the length of each time series. If the time series have different lengths, the return
        value should be a list that contains the length of each sequence. If all sequences are of equal length,
        this should return an `int`.
        """
        return self.parent.seq_len if self.parent is not None else None

    @property
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        """
        Number of features of each datapoint. This can also be a tuple if the data has more than one feature dimension.
        """
        return self.parent.num_features if self.parent is not None else None
