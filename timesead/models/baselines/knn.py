from typing import Optional, Tuple, Union

import torch
from pyod.models.knn import KNN
from numpy.lib.stride_tricks import sliding_window_view

from ..common import AnomalyDetector
from ...data.statistics import get_data_all

class KNNAD(AnomalyDetector):
    def __init__(
        self,
        n_neighbors: int = 5,
        method: str = "largest",
        window_size: int = 12,
        stride: int = 1,
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
            window_size[int]: Time frame window size to use for the model
            stride[int]: Stride value for sliding window
        """
        super(KNNAD, self).__init__()

        self.n_neighbors = n_neighbors
        self.method = method
        self.window_size = window_size
        self.stride = stride
        self.input_shape = input_shape

        self.model = KNN(n_neighbors=self.n_neighbors,
                         method=self.method)

    def _preprocess_data(self, input: torch.tensor) -> torch.tensor:
        # Converts data to window_size chunks
        flat_shape = (input.shape[0] - (self.window_size - 1), -1)
        slides = sliding_window_view(input, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
        return torch.from_numpy(slides)


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        data = self._preprocess_data(get_data_all(dataset.dataset))
        self.model.fit(data)


    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        batch_input = inputs[0]
        # Convert input to (B, T, D) dimension
        if self.input_shape[0] == "t":
            batch_input = batch_input.permute(1, 0, 2)

        # Get the final window for each batch
        data = batch_input[:, -self.window_size:, :]
        # Reshape to match sliding_window_view output
        data = data.permute(0, 2, 1).reshape(data.shape[0], -1)

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
