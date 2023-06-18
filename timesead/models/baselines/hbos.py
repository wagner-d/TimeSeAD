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
        data = get_data_all(dataset.dataset)

        self.model.fit(data)


    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        # TODO: test for input shape (T, B, D)
        data = inputs[0]
        pred_scores = list(map(self.model.decision_function, data))
        # Converting list of np arrays to a single np array and then converting to tensor is faster
        # as per pytorch user warning
        pred_scores = torch.tensor(np.array(pred_scores))

        return pred_scores[:, -1] if self.input_shape[0] == "b" else pred_scores[-1]


    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        # Input of shape (B, T) or (T, B), output of shape (B)
        (target,) = targets

        # Just return the last label of the window
        return target[:, -1] if self.input_shape[0] == "b" else target[-1]
