from typing import Tuple, Union, List

import torch

from .transform_base import Transform
from ..dataset import BaseTSDataset


def _handle_negative_index(index: int, length: int):
    """

    :param index:
    :param length:
    :return:
    """
    if length < 0:
        raise ValueError("Sequence length must be non-negative!")

    if index > length or index < -length:
        raise IndexError

    return length + index if index < 0 else index


class DatasetSource(Transform):
    """
    This acts as a source :class:`~timesead.data.transforms.Transform` (meaning it has no parent) that simply returns
    sequences from a given dataset.
    It can be constrained to return only a specific part of the data.
    """
    def __init__(self, dataset: BaseTSDataset, start: Union[int, List[int]] = None,
                 end: Union[int, List[int]] = None, axis: str = 'batch'):
        """

        :param dataset: The dataset from which to take points.
        :param start: Start index for this dataset. Please see below for a more detailed explanation.
        :param end: End index for this dataset (exclusive). Please see below for a more detailed explanation.
        :param axis: Can be either 'batch' or 'time'. In 'batch' mode, this simply returns only the sequences indexed
            from `start` to `end`. 'time' mode is used for datasets that contain only one long time series. That time
            series will be cut according to `start` and `end`.
        """
        super(DatasetSource, self).__init__(None)

        self.dataset = dataset
        self.axis = axis if axis == 'time' else 'batch'

        data_len = len(dataset)
        if start is None:
            start = 0
        if end is None:
            end = data_len

        if self.axis == 'time':
            data_len = dataset.seq_len
            if isinstance(data_len, int):
                data_len = [data_len] * len(dataset)

            if isinstance(start, int):
                start = [start] * len(dataset)

            if isinstance(end, int):
                end = [end] * len(dataset)

            self.start = [_handle_negative_index(i, l) for i, l in zip(start, data_len)]
            self.end = [_handle_negative_index(i, l) for i, l in zip(end, data_len)]

            if any(s > e for s, e in zip(self.start, self.end)):
                raise ValueError("start must be smaller or equal to end!")
        else:
            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError("start and end must be integers! Lists not allowed when splitting along the batch axis")

            self.start = _handle_negative_index(start, data_len)
            self.end = _handle_negative_index(end, data_len)

            if self.start > self.end:
                raise ValueError("start must be smaller or equal to end!")

    def _get_datapoint_impl(self, item) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if self.axis == 'batch':
            index = self.start + item
            return self.dataset[index]

        # Slice in time dimension
        inputs, targets = self.dataset[item]
        inputs = tuple(inp[self.start[item]:self.end[item]] for inp in inputs)
        targets = tuple(target[self.start[item]:self.end[item]] for target in targets)

        return inputs, targets

    def __len__(self):
        return len(self.dataset) if self.axis == 'time' else (self.end - self.start)

    @property
    def seq_len(self):
        return self.dataset.seq_len if self.axis == 'batch' else [(end - start) for start, end in zip(self.start, self.end)]

    @property
    def num_features(self):
        return self.dataset.num_features


def make_dataset_split(dataset: BaseTSDataset, *splits: float, axis: str = 'batch'):
    r"""
    Create :class:`DatasetSource`\s for different parts of a given dataset.

    :param dataset: The dataset, for which the split should be done.
    :param splits: This should be the percentages of the dataset in each split. Will be normalized to 100%.
    :param axis: The axis along which to split the dataset. Please see :class:`DatasetSource` for a more detailed
        explanation.
    :return: This will return a generator that yields :class:`DatasetSource`\s according to the specified splits.
    """
    axis = axis if axis == 'time' else 'batch'

    # Compute relative split percentages
    percent = torch.tensor(splits, dtype=torch.float64)
    percent /= torch.sum(percent)

    # Translate percentages into index ranges
    if axis == 'batch':
        data_len = len(dataset)
        lengths = torch.floor(percent * data_len).to(torch.int64)
        rest = data_len - torch.sum(lengths).item()
        rest = torch.tensor([1]*rest + [0]*(len(lengths) - rest))
        lengths += rest

        cum_bound = 0
        for l in lengths:
            start = cum_bound
            cum_bound += l.item()
            yield DatasetSource(dataset, start, cum_bound, axis=axis)
    else:
        data_len = dataset.seq_len
        if isinstance(data_len, int):
            data_len = [data_len] * len(dataset)

        starts, ends = [[] for _ in percent], [[] for _ in percent]
        for data_l in data_len:
            lengths = torch.floor(percent * data_l).to(torch.int64)
            rest = data_l - torch.sum(lengths).item()
            rest = torch.tensor([1] * rest + [0] * (len(lengths) - rest))
            lengths += rest

            cum_bound = 0
            for i, l in enumerate(lengths):
                start = cum_bound
                cum_bound += l.item()
                starts[i].append(start)
                ends[i].append(cum_bound)

        for start, end in zip(starts, ends):
            yield DatasetSource(dataset, start, end, axis=axis)
