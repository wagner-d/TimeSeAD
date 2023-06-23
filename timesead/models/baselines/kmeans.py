from typing import Tuple
import torch
from sklearn.cluster import MiniBatchKMeans

from ..common import AnomalyDetector

class KMeansAD(AnomalyDetector):
    def __init__(self, k: int, batch_size: int, input_shape: str = "btf") -> None:
        """
        KMeans Anomaly Detector
            Compute anomalies using the K-Means algorithm.
            Anomaly score is computed as distance from the matched cluster center.

        Args:
            k[int]: K value for K-Means algorithm
            batch_size[int]: Batch size for the mini batches
        """
        super(KMeansAD, self).__init__()

        self.k = k
        self.batch_size = batch_size
        self.model = MiniBatchKMeans(n_clusters=k, batch_size=self.batch_size)
        self.input_shape = input_shape


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        for (b_inputs, b_targets) in dataset:
            data = b_inputs[0]
            batch_size, window_size, n_features = data.shape
            self.window_size = window_size
            data = data.reshape(batch_size, window_size*n_features)
            self.model.partial_fit(data)


    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        batch_input = inputs[0]
        # Convert input to (B, T, D) dimension
        if self.input_shape[0] == "t":
            batch_input = batch_input.permute(1, 0, 2)

        # Get the final window for each batch
        data = batch_input[:, -self.window_size:, :]
        data = data.reshape(data.shape[0], -1)
        clusters = self.model.predict(data)
        diffs = torch.linalg.norm(data - self.model.cluster_centers_[clusters], axis=1)
        return diffs


    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        # Input of shape (B, T) or (T, B), output of shape (B)
        (target,) = targets

        # Just return the last label of the window
        return target[:, -1] if self.input_shape[0] == "b" else target[-1]
