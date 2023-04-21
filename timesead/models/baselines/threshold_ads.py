from typing import Optional, Tuple, Union

import torch

from ..common import AnomalyDetector
from ...data.statistics import compute_feature_statistics

class Base_ThresholdAD(AnomalyDetector):
    def __init__(
        self,
        first_diffs: bool,
        cum_method: str,
        feature_index: Optional[int],
        input_shape: str = "btf",
    ) -> None:
        """
        A Basis Threshold Anomaly Detector
            A simlpe anomaly detector that is equals zero for all data within the given Threshold and else equals
            the distance to the given lower/upper thr of training value.

        Args:
            first_diffs[bool]: Flag, if instead of raw values first difference should be used.
            cum_method[str]: One of {'mean', 'max'}. Accumulation method over feature dimension. Note that when
                `feature_index` is not None. Then accumulation method will be ignored.
            feature_index[optional[int]]: Take scores for specific feature. If none, above accumulation rule will
                be used.
        """

        super(Base_ThresholdAD, self).__init__()

        self.first_diffs = first_diffs
        if cum_method.lower() not in ["mean", "max"]:
            raise ValueError("`cum_method` must be one of {'mean','max'}")
        self.cum_method = cum_method.lower()
        self.feature = feature_index
        self.input_shape = input_shape
        self.lower_thresh= None
        self.upper_thresh = None

    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        # Input of shape (B, T, D), output of shape (B,)

        # Extract time dimension
        t_dim = 1 if self.input_shape[0] == "b" else 0

        if self.first_diffs:  # Take fd over time dim
            X = torch.diff(inputs[0], dim=t_dim)
            # Since diff "deletes" first observation in batch
            # we dublicate first fd
            X = torch.cat(
                (X[:, 0:1, :] if t_dim == 1 else X[0:1, :, :], X),
                dim=t_dim,
            )
        else:
            X = inputs[0]

        # Calculate scores
        scores = torch.relu(X - self.lower_thresh) + torch.relu(X - self.upper_thresh)
        
        if self.feature:
            scores = scores[:,:,self.feature]
        else:
            if self.cum_method == "max":
                scores = scores.max(dim=-1).values
            else:
                scores = scores.mean(dim=-1)

        return scores[:, -1] if t_dim == 1 else scores[-1, :]

    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        # Input of shape (B, T) or (T, B), output of shape (B)
        (target,) = targets

        # Just return the last label of the window
        return target[:, -1] if self.input_shape[0] == "b" else target[-1]

class OOSAnomalyDetector(Base_ThresholdAD):
    def __init__(
        self,
        first_diffs: bool = False,
        cum_method: str = "mean",
        feature_index: Optional[int] = None,
        *args, **kwargs,
    ) -> None:
        """
        Out of Support Anomaly Detector
            A simlpe anomaly detector that is equals zero for all data within the data support of the training data
            and else equals the distance to the min/max of training value.

        Args:
            first_diffs[bool]: Flag, if instead of raw values first difference should be used. Default is False.
            cum_method[str]: One of {'mean', 'max'}. Accumulation method over feature dimension. Note that when
                `feature_index` is not None. Then accumulation method will be ignored. Default is 'mean'.
            feature_index[optional[int]]: Take scores for specific feature. If none, above accumulation rule will
                be used. Default is None.
        """
        super(OOSAnomalyDetector, self).__init__(first_diffs, cum_method, feature_index, *args, **kwargs)

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        
        # Get minumum and maximum from train data as lower and upper thresholds
        if (self.lower_thresh is None) or (self.upper_thresh is None):
            *_, self.lower_thresh, self.upper_thresh = compute_feature_statistics(
                dataset.dataset, take_fd=self.first_diffs
            )
        
        # Send to device
        self.lower_thresh = self.lower_thresh.to(self.dummy.device)
        self.upper_thresh = self.upper_thresh.to(self.dummy.device)

class IQRAnomalyDetector(Base_ThresholdAD):
    def __init__(
        self,
        std_factor: float = 2.58,
        first_diffs: bool = False,
        cum_method: str = "mean",
        feature_index: Optional[int] = None,
        *args, **kwargs,
    ) -> None:
        """
        Interquantile Range Anomaly Detector
            A simlpe anomaly detector that is equals zero for all data within some interquantile range of the
            normal training data and else equals the distance to nearest quantile border.

        Args:
            std_factor[float]: float, that gives the width of the IQR. Default is 2.58 (=99.5% normal quantile).
            first_diffs[bool]: Flag, if instead of raw values first difference should be used. Default is False.
            cum_method[str]: One of {'mean', 'max'}. Accumulation method over feature dimension. Note that when
                `feature_index` is not None. Then accumulation method will be ignored. Default is 'mean'.
            feature_index[optional[int]]: Take scores for specific feature. If none, above accumulation rule will
                be used. Default is None.
        """
        self.std_factor = std_factor

        super(IQRAnomalyDetector, self).__init__(first_diffs, cum_method, feature_index, *args, **kwargs)

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:

        if (self.lower_thresh is None) or (self.upper_thresh is None):
            mean, std, *_ = compute_feature_statistics(
                dataset.dataset, take_fd=self.first_diffs
            )
            self.lower_thresh = mean - self.std_factor * std
            self.upper_thresh = mean + self.std_factor * std
        
        # Send to device
        self.lower_thresh = self.lower_thresh.to(self.dummy.device)
        self.upper_thresh = self.upper_thresh.to(self.dummy.device)