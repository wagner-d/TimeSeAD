from typing import Optional, Tuple

import torch
import numpy as np
from pyod.models.hbos import HBOS

from ..common import AnomalyDetector
from ...data.statistics import get_data_all

class HBOSAD(AnomalyDetector):
    def __init__(
        self,
        n_bins: Optional[int] = 10,
        alpha: float = 0.1,
        bin_tol: float = 0.5,
        input_shape: str = "btf",
    ) -> None:
        """
        Histogram Based Outlier Score
            The method assumes feature independence and calculates the degree of outlyingness by
            building histograms. See [Goldstein2012] for details.

            Implementation derived from https://github.com/HPI-Information-Systems/TimeEval-algorithms

        .. [Goldstein2012] Markus Goldstein and Andreas Dengel. 2012.
            Histogrambased Outlier Score (HBOS): A fast Unsupervised Anomaly Detection Algorithm.
            In Proceedings of the German Conference on Artificial Intelligence Poster and Demo Track (KI), 59-63.

        Args:
            n_bins[Optional[int]]: The number of bins. Set to None for automatic selection.
            alpha[float]: The regularizer for preventing overflow.
            bin_tol[float]: The parameter to decide the flexibility while dealing with
                the samples falling outside the bins.
        """
        super(HBOSAD, self).__init__()

        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_tol = bin_tol
        self.input_shape = input_shape

        self.model = HBOS(
            n_bins=self.n_bins or "auto",
            alpha=self.alpha,
            tol=self.bin_tol
        )


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        # Merge all batches as KNN can't do batch processing
        data_full = []
        for (b_inputs, b_targets) in dataset:
            data = b_inputs[0]
            batch_size, window_size, n_features = data.shape
            self.window_size = window_size
            data = data.reshape(batch_size, window_size*n_features)
            data_full.append(data)
        data_full = torch.cat(data_full)

        self.model.fit(data_full)


    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        batch_input = inputs[0]
        # Convert input to (B, T, D) dimension
        if self.input_shape[0] == "t":
            batch_input = batch_input.permute(1, 0, 2)

        if not hasattr(self, 'window_size'):
            raise RuntimeError('Run "fit" function before trying to compute_anomaly_score')
        # Get the final window for each batch
        data = batch_input[:, -self.window_size:, :]
        data = data.reshape(data.shape[0], -1)

        scores = self.model.decision_function(data)
        scores = torch.tensor(scores)

        return scores


    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        # Input of shape (B, T) or (T, B), output of shape (B)
        (target,) = targets

        # Just return the last label of the window
        return target[:, -1] if self.input_shape[0] == "b" else target[-1]
