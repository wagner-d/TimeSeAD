from typing import Optional, Tuple, Union

import torch
import numpy
from pyod.models.knn import KNN

from ..common import AnomalyDetector
from ...data.statistics import get_data_all

class KNNAD(AnomalyDetector):
    def __init__(
        self,
        n_neighbors: int = 5,
        method: str = "largest",
        input_shape: str = "btf",
    ) -> None:
        # TODO: Should first diffs be impelemented?
        """
        KNN Anomaly Detector
            The distance to the k'th nearest neighbor is used as the metric for Anomaly detection.
            Implementation derived from https://github.com/HPI-Information-Systems/TimeEval-algorithms

        Args:
            n_neightbors[int]: Number of neighbors for KNN algorithm
            method[str]: How to calculate the score for KNN. One of {'largest', 'mean', 'median'}.
        """
        super(KNNAD, self).__init__()

        self.n_neighbors = n_neighbors
        self.method = method
        self.input_shape = input_shape

        self.model = KNN(n_neighbors=self.n_neighbors,
                         method=self.method)


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
        pred_scores = torch.tensor(numpy.array(pred_scores))

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