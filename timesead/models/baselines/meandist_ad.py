from typing import Optional, Tuple, Union

import torch

from ..common import AnomalyDetector
from ...data.statistics import compute_feature_statistics, compute_whiten_matrix

class WMDAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        first_diffs: bool = False,
        cum_method: str = "max",
        feature_index: Optional[int] = None,
        full_cov: bool = False,
        input_shape: str = "btf",
    ) -> None:
        """
        Weighted Mean Distance Anomaly Detector
            A simlpe anomaly detector that outputs the distance to the mean (mean from training data) weighted by
            each feature standard deviation.

        Args:
            first_diffs[bool]: Flag, if instead of raw values first difference should be used. Default is False.
            cum_method[str]: One of {'mean', 'max'}. Accumulation method over feature dimension. Note that when
                `feature_index` is not None. Then accumulation method will be ignored. Default is 'max'.
            feature_index[optional[int]]: Take scores for specific feature. If none, above accumulation rule will
                be used. Default is None.
            full_cov[bool]: Take full covariance matrix to weight diviation from the mean. Default is False.
        """
        super(WMDAnomalyDetector, self).__init__()

        self.first_diffs = first_diffs
        if cum_method.lower() not in ["mean", "max"]:
            raise ValueError("`cum_method` must be one of {'mean','max'}")
        self.cum_method = cum_method.lower()
        self.feature = feature_index
        self.full_cov = full_cov
        self.input_shape = input_shape
        self.mean = None
        self.inv_std = None

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:

        if (self.mean is None) or (self.inv_std is None):
            self.mean, std, *_ = compute_feature_statistics(
                dataset.dataset, take_fd=self.first_diffs
            )
            if self.full_cov:
                self.inv_std = compute_whiten_matrix(
                    dataset.dataset, take_fd=self.first_diffs
                )
            else:
                # Deal with constant features
                std[std < torch.finfo(torch.float32).eps] = 1.
                # Take inverse
                self.inv_std = torch.diag(1/std)

        # send to device
        self.mean = self.mean.to(self.dummy.device)
        self.inv_std = self.inv_std.to(self.dummy.device)

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

        # Calculate scores: Absolute difference from mean weighted by std
        scores = torch.abs((X - self.mean) @ self.inv_std)
        
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