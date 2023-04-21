import functools
from typing import Tuple

import torch

from .transform_base import Transform
from ...utils.utils import ceil_div, getitem


class SubsampleTransform(Transform):
    """
    Subsample sequences by a specified factor. `subsampling_factor` consecutive datapoints in a sequence will be
    aggregated into one point using the `aggregation` function.
    """
    def __init__(self, parent: Transform, subsampling_factor: int, aggregation: str = 'first'):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param subsampling_factor: This specifies the number of consecutive data points that will be aggregated.
        :param aggregation: The function that should be applied to aggregate a window of data points.
           Can be either 'mean', 'last' or 'first'.
        """
        super(SubsampleTransform, self).__init__(parent)

        self.subsampling_factor = subsampling_factor
        if aggregation == 'mean':
            self.aggregate_fn = functools.partial(torch.mean, dim=1)
        if aggregation == 'last':
            self.aggregate_fn = functools.partial(getitem, item=(slice(None), -1))
        else:  # 'first'
            self.aggregate_fn = functools.partial(getitem, item=(slice(None), 0))

    def _process_tensor(self, inp: torch.Tensor) -> torch.Tensor:
        # Input has shape (T, ...). Reshape it to (T//subsampling_factor, subsampling_factor, ...) and apply
        # aggregate on the 2nd axis. We might need to add padding at the end for this to work
        inp_shape = inp.shape
        new_t, rest = divmod(inp_shape[0], self.subsampling_factor)
        if rest > 0:
            # Add padding. We pad with the result of aggregating the last (incomplete) window
            pad_value = self.aggregate_fn(inp[new_t * self.subsampling_factor:inp_shape[0]].unsqueeze(0))
            inp = torch.cat([inp] + (self.subsampling_factor - rest) * [pad_value])
            new_t += 1

        inp = inp.view(new_t, self.subsampling_factor, *inp_shape[1:])
        return self.aggregate_fn(inp)

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)

        inputs = tuple(self._process_tensor(inp) for inp in inputs)
        targets = tuple(self._process_tensor(tar) for tar in targets)

        return inputs, targets

    @property
    def seq_len(self):
        old_len = self.parent.seq_len
        if old_len is None:
            return None

        if isinstance(old_len, int):
            return ceil_div(old_len, self.subsampling_factor)

        return [ceil_div(old_l, self.subsampling_factor) for old_l in old_len]


class CacheTransform(Transform):
    """
    Caches the results from a previous :class:`~timesead.data.transforms.Transform` in memory so that expensive
    calculations do not have to be recomputed.
    """
    def __init__(self, parent: Transform):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        """
        super(CacheTransform, self).__init__(parent)

        self.cache = {}

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if item in self.cache:
            return self.cache[item]

        inputs, targets = self.parent.get_datapoint(item)
        self.cache[item] = (inputs, targets)

        return inputs, targets


class LimitTransform(Transform):
    """
    Limits the amount of data points returned.
    """
    def __init__(self, parent: Transform, count: int):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param count: The max number of sequences that should be returned by this
            :class:`~timesead.data.transforms.Transform`.
        """
        super(LimitTransform, self).__init__(parent)
        self.max_count = count

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if item >= self.max_count:
            raise IndexError

        return self.parent.get_datapoint(item)

    def __len__(self):
        if len(self.parent) is not None:
            return min(self.max_count, len(self.parent))

        return None
