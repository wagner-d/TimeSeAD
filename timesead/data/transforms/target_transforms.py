from typing import Tuple, Optional, List, Any, Union

import torch

from .transform_base import Transform
from .window_transform import WindowTransform


class ReconstructionTargetTransform(Transform):
    """
    Adds the current inputs as targets for reconstruction objectives.
    """
    def __init__(self, parent: Transform, replace_labels: bool = False):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            transform.
        :param replace_labels: Whether the original labels should be replaced by the reconstruction target.
           If `False`, the reconstruction target will be added to the tuple of original labels.
        """
        super(ReconstructionTargetTransform, self).__init__(parent)
        self.replace_labels = replace_labels

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)

        if self.replace_labels:
            return inputs, inputs

        return inputs, targets + inputs


class OneVsRestTargetTransform(Transform):
    """
    Transforms multi-class labels into binary labels for anomaly detection.
    "Normal" data points will have label 0, others will have label 1.
    """
    def __init__(self, parent: Transform, normal_class: Optional[Any] = None, anomalous_class: Optional[Any] = None,
                 replace_labels: bool = False):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param normal_class: The input class label that should be considered normal and will have label 0 in the output.
        :param anomalous_class: You can also specify an anomalous class that will have label 1.
            All other labels will be transformed to 0. Note that you cannot specify both `normal_class` and
            `anomalous_class`.
        :param replace_labels: Whether the original labels should be replaced by the
            :class:`~timesead.data.transforms.Transform`.
            If `False`, the additional labels will be added to the tuple of original labels.
        """
        super(OneVsRestTargetTransform, self).__init__(parent)
        self.replace_labels = replace_labels

        if normal_class is None and anomalous_class is None:
            raise ValueError('Must set either normal_class or anomalous_class!')
        if normal_class is not None and anomalous_class is not None:
            raise ValueError('Cannot specify both normal_class and anomalous_class!')

        self.normal_class = normal_class
        self.anomalous_class = anomalous_class

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)

        if self.normal_class is not None:
            new_targets = tuple(torch.where(target == self.normal_class, 0, 1) for target in targets)
        elif self.anomalous_class is not None:
            new_targets = tuple(torch.where(target == self.anomalous_class, 1, 0) for target in targets)
        else:
            new_targets = targets

        if self.replace_labels:
            return inputs, new_targets

        return inputs, targets + new_targets


class PredictionTargetTransform(WindowTransform):
    """
    Adds the last `prediction_window` points from the current inputs as targets for prediction objectives.
    """
    def __init__(self, parent: Transform, window_size: int, prediction_horizon: int, replace_labels: bool = False,
                 step_size: int = 1, reverse: bool = False):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param prediction_horizon: Number of datapoints that should be predicted.
        :param replace_labels: Whether the original labels should be replaced by the prediction target.
           If `False`, the prediction target will be added to the tuple of original labels.
        """
        super(PredictionTargetTransform, self).__init__(parent, window_size + prediction_horizon, step_size, reverse)

        self.input_window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.replace_labels = replace_labels

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = super(PredictionTargetTransform, self)._get_datapoint_impl(item)

        new_inputs = tuple(inp[:-self.prediction_horizon] for inp in inputs)
        new_targets = tuple(inp[-self.prediction_horizon:] for inp in inputs)

        if self.replace_labels:
            return new_inputs, new_targets

        targets = tuple(target[-self.prediction_horizon:] for target in targets)
        return new_inputs, targets + new_targets

    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self.input_window_size


class OverlapPredictionTargetTransform(Transform):
    """
    Adds the sequence shifted by offset as the target.
    """
    def __init__(self, parent: Transform, offset: int, replace_labels: bool = False):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param offset: Number of steps ahead that should be predicted.
        :param replace_labels: Whether the original labels should be replaced by the prediction target.
           If `False`, the prediction target will be added to the tuple of original labels.
        """
        super(OverlapPredictionTargetTransform, self).__init__(parent)
        self.offset = offset
        self.replace_labels = replace_labels

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)

        new_inputs = tuple(inp[:-self.offset] for inp in inputs)
        new_targets = tuple(inp[self.offset:] for inp in inputs)

        if self.replace_labels:
            return new_inputs, new_targets

        targets = tuple(target[self.offset:] for target in targets)
        return new_inputs, targets + new_targets

    @property
    def seq_len(self) -> Union[int, List[int]]:
        parent_seq_len = self.parent.seq_len
        if isinstance(parent_seq_len, int):
            return parent_seq_len - self.offset

        return [slen - self.offset for slen in parent_seq_len]
