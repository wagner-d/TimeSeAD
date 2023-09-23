from typing import Optional, Tuple, Union

import torch
from pyod.models.knn import KNN

from ..common import AnomalyDetector

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
