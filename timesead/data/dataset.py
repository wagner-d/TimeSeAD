import abc
import collections.abc
import functools
from typing import Tuple, Union, Callable, Any, Dict, List

import torch
from torch._six import string_classes
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format


class BaseTSDataset(abc.ABC, Dataset):
    """
    Base class for all time-series datasets in TimeSeAD. Implementing the members in this abstract class provides the
    data pipeline system with the necessary information to process the data correctly.
    """
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        This should return the number of independent time series in the dataset
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def seq_len(self) -> Union[int, List[int]]:
        """
        This should return the length of each time series. If the time series have different lengths, the return
        value should be a list that contains the length of each sequence. If all sequences are of equal length,
        this should return an int.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        """
        Number of features of each datapoint. This can also be a tuple if the data has more than one feature dimension.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        """
        Return the default pipeline for this dataset that is used if the user does not specify a different pipeline.
        This must be a dict of the form::

            {
                '<name>': {'class': '<name-of-transform-class>', 'args': {'<args-for-constructor>', ...}},
                ...
            }
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_feature_names() -> List[str]:
        """
        Return names for the features in the order they are present in the data tensors.

        :return: A list of strings with names for each feature.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Access the timeseries at position `index` and its corresponding label sequence. A call to this function should
        return a single time series that was sampled independently of the other time series in this dataset.

        :param index: The zero-based index of the time series to retrieve.
        :return: A tuple `(inputs, targets)`, where inputs is again a tuple of :class:`~torch.Tensor`\s with shape
            `(T, D*)`, where `D*` can very between the tensors. `targets` contains labels for the time series as tensors
            of shape `(T,)`.
        """
        raise NotImplementedError


def __default_collate(batch, batch_dim: int = 0):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, dim=batch_dim, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return __default_collate([torch.as_tensor(b) for b in batch], batch_dim=batch_dim)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: __default_collate([d[key] for d in batch], batch_dim=batch_dim) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(__default_collate(samples, batch_dim=batch_dim) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [__default_collate(samples, batch_dim=batch_dim) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_fn(batch_dim: int) -> Callable:
    """
    Puts each data field into a tensor with outer dimension batch size.

    This was largely copied from PyTorch's :func:`~torch.utils.data._utils.default_collate` function except that it
    allows concatenating :class:`~torch.Tensor`\s along an arbitrary dimension instead of always stacking along the
    first dimension.

    :param batch_dim: The index of the dimension along which to stack the elements in the batch.
    :return: Batched tensors where elements have been stacked along the `batch_dim` dimension.
    """
    return functools.partial(__default_collate, batch_dim=batch_dim)
