import abc
from collections import Counter
from typing import Callable, Tuple

import numpy as np
import torch

from .transform_base import Transform
from ...utils.utils import generate_intervals


class InjectArtificialAnomaliesTransform(Transform):
    """
    This Transform injects anomalies into the dataset.

    It expects the get_datapoint method of its parent to return a tuple of tuples of length 1.
    """
    def __init__(self, parent: Transform, n: int, min_length: int = 1, max_length: int = 1):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param n: Number of anomalies to insert.
        :param min_length: Minimum length of anomalies.
        :param max_length: Maximum length of anomalies.
        """

        super(InjectArtificialAnomaliesTransform, self).__init__(parent)

        self._sample_intervals(n, min_length, max_length)

    def _sample_intervals(self, n: int, min_length: int, max_length: int):

        intervals_per_time_series = self._compute_intervals_per_time_series(n, min_length)

        # Collect the indices of all anomalies in a dictionary for each time series that has anomalies injected
        self.indices = {}

        for time_series, n_intervals in intervals_per_time_series.most_common():

            # Compute the boundaries of intervals that anomalies are to be injected in
            self.indices[time_series] = {
                'intervals': generate_intervals(n_intervals, min_length, max_length,
                                                self.parent.get_datapoint(time_series)[0][0].shape[0])
            }

            # Compute the anomalies for the intervals
            self.indices[time_series]['values'] = [
                self._inject_anomaly(self.parent.get_datapoint(time_series)[0][0][left:right]) for left, right in self.indices[time_series]['intervals']
            ]

    def _compute_intervals_per_time_series(self, n: int, min_length: int):

        sizes = [self.parent.get_datapoint(idx)[0][0].shape[0] for idx in range(len(self))]

        # Check if it is possible to sample n non overlapping windows from the dataset
        assert sum([size // min_length for size in sizes]) >= n

        candidate_time_series = {idx: sizes[idx] // min_length for idx in range(len(self)) if sizes[idx] // min_length}

        datapoint_indices = []

        for i in range(n):

            candidates = list(candidate_time_series.keys())
            total_size = sum(candidate_time_series.values())
            probabilities = [candidate_time_series[c] / total_size for c in candidates]

            time_series_index = np.random.choice(candidates, p=probabilities)

            if candidate_time_series[time_series_index] == 1:
                del candidate_time_series[time_series_index]
            else:
                candidate_time_series[time_series_index] -= 1

            datapoint_indices.append(time_series_index)

        intervals_per_time_series = Counter(datapoint_indices)

        return intervals_per_time_series

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:

        inputs, targets = self.parent.get_datapoint(item)

        if item in self.indices:

            for (l, r), value in zip(self.indices[item]['intervals'], self.indices[item]['values']):

                inputs[0][l:r]  = value
                targets[0][l:r] = 1

        return inputs, targets

    @abc.abstractmethod
    def _inject_anomaly(self, interval: torch.Tensor) -> torch.Tensor:
        """Injects an anomaly into an interval.

        :param interval: Interval in a time series of the dataset.
        :return: Interval of the same length to replace the input interval in the dataset.
        """
        raise NotImplementedError


class InjectIndependentArtificialAnomaliesTransform(InjectArtificialAnomaliesTransform):
    """
    Transform that injects anomalies, that only depend on the anomaly interval.
    """
    def __init__(self, parent: Transform, anomaly_fn: Callable, n: int, min_length: int = 1, max_length: int = 1):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param anomaly_fn: Callable that adds an anomaly to an interval and returns a :class:`torch.Tensor` of the same
            size as its input.
        :param n: Number of anomalies to insert.
        :param min_length: Minimum length of anomalies.
        :param max_length: Maximum length of anomalies.
        """

        self.anomaly = anomaly_fn

        super(InjectIndependentArtificialAnomaliesTransform, self).__init__(parent, n, min_length, max_length)

    def _inject_anomaly(self, interval: torch.Tensor, index: int = -1, left_boundary: int = -1, right_boundary: int = -1) -> torch.Tensor:
        return self.anomaly(interval)


class InjectWindowsArtificialAnomaliesTransform(InjectArtificialAnomaliesTransform):
    """
    :class:`~timesead.data.transforms.Transform` that inject windows from somewhere else in the dataset as anomalies.
    """
    def __init__(self, parent: Transform, mask_fn: Callable, n: int, min_length: int = 1, max_length: int = 1):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param mask_fn: Callable that computes a mask to the features of an interval.
        :param n: Number of anomalies to insert.
        :param min_length: Minimum length of anomalies.
        :param max_length: Maximum length of anomalies.
        """

        self.mask = mask_fn

        super(InjectWindowsArtificialAnomaliesTransform, self).__init__(parent, n, min_length, max_length)

    def _sample_intervals(self, n: int, min_length: int, max_length: int):

        intervals_per_time_series = self._compute_intervals_per_time_series(2*n, min_length)

        all_indices = list(intervals_per_time_series.elements())

        np.random.shuffle(all_indices)

        # Collect all the indices of time series in the dataset, that anomalies are to be inserted in
        time_series_containing_anomalies = Counter(all_indices[:n])

        # self.reference_time_series is a sorted list of indices of time series in the dataset,
        # where ich index is repeated as often as how many reference intervals it contains
        # self.referent_indices is a list containing the relative indices of intervals in the time series
        # i.e. reference interval i is the self.reference_indices[i]th interval in self.reference_time_series[i]
        self.reference_time_series = sorted(list(Counter(all_indices[n:]).elements()))
        self.reference_indices     = []

        # self.intervals contains all intervals for a time series in the dataset
        self.intervals = {}

        self.indices = {}

        # Generate the intervals for each time series
        for time_series in sorted(intervals_per_time_series.keys()):

            n_intervals = intervals_per_time_series[time_series]

            intervals = generate_intervals(n_intervals,
                                           min_length,
                                           max_length,
                                           self.parent.get_datapoint(time_series)[0][0].shape[0])

            # Collect the indices of intervals where anomalies are to be inserted
            if time_series in time_series_containing_anomalies.keys():
                anomaly_indices = sorted(np.random.choice(list(range(len(intervals))),
                                                          size=time_series_containing_anomalies[time_series],
                                                          replace=False))
            else:
                anomaly_indices = []

            self.intervals[time_series] = intervals

            self.reference_indices.extend([idx for idx in range(len(intervals)) if idx not in anomaly_indices])

            if anomaly_indices:
                self.indices[time_series] = {
                    'intervals': [intervals[idx] for idx in anomaly_indices],
                    'values': []
                }

        for time_series in self.indices.keys():

            for idx, (left_boundary, right_boundary) in enumerate(self.indices[time_series]['intervals']):

                anomaly = self._inject_anomaly(self.parent.get_datapoint(time_series)[0][0][left_boundary:right_boundary])

                # If the reference interval is smaller than the interval that the anomaly is to be inserted in,
                # the shape is different and thus has to be updated
                self.indices[time_series]['intervals'][idx] = (left_boundary, left_boundary + anomaly.shape[0])

                self.indices[time_series]['values'].append(anomaly)

    def _inject_anomaly(self, interval: torch.Tensor) -> torch.Tensor:

        # Draw a random reference interval
        reference_index = np.random.choice(list(range(len(self.reference_indices))))

        # Get the index of the time series reference interval is in
        # and the index in the list of intervals in that time series
        reference_time_series = self.reference_time_series[reference_index]
        reference_interval    = self.reference_indices[reference_index]

        # Remove elements from reference lists
        self.reference_time_series = self.reference_time_series[:reference_index] + self.reference_time_series[reference_index + 1:]
        self.reference_indices     = self.reference_indices[:reference_index] + self.reference_indices[reference_index + 1:]

        # Adjust the size of the intervals to match each other

        left_boundary, right_boundary = self.intervals[reference_time_series][reference_interval]

        if interval.shape[0] < right_boundary - left_boundary:
            right_boundary = left_boundary + interval.shape[0]

        # Extract the anomaly
        anomaly = self.parent.get_datapoint(reference_index)[0][0][left_boundary:right_boundary]

        mask = self.mask(anomaly)

        # Apply mask and return constructed anomaly
        return torch.mul(interval[:right_boundary-left_boundary], mask) + torch.mul(anomaly, 1 - mask)
