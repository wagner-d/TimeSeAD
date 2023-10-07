from typing import Optional, Tuple, Union

import torch
from pyod.models.iforest import IForest

from ..common import AnomalyDetector

class IForestAD(AnomalyDetector):
    def __init__(
        self,
        n_trees: int = 100,
        max_samples: Optional[float] = None,
        max_features: float = 1.,
        bootstrap: bool = False,
        input_shape: str = "btf",
    ) -> None:
        """
        Isolation Forest Anomaly Detector
            The Isolation Forest 'isolates' observations by randomly selecting a feature and then randomly
            selecting a split value between the maximum and minimum of the selected feature.
            See [Liu2008] for more details.

            Implementation derived from https://github.com/HPI-Information-Systems/TimeEval-algorithms

        .. [Liu2008] F. T. Liu, K. M. Ting and Z. -H. Zhou,
            "Isolation Forest,"
            2008 Eighth IEEE International Conference on Data Mining, Pisa, Italy,
            2008, pp. 413-422, doi: 10.1109/ICDM.2008.17.

        Args:
            n_trees[int]: Number of trees in the forest.
            max_samples[Optional[float]]: The number of samples to train each tree.
            max_features[float]: Percent of features to draw from to train each tree.
            bootstrap[bool]: If True, individual trees are fit on random subsets of the training data
                sampled with replacement. Else sampling is without replacement.
        """
        super(IForestAD, self).__init__()

        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.input_shape = input_shape

        self.model = IForest(
            n_estimators=self.n_trees,
            max_samples=self.max_samples or "auto",
            max_features=self.max_features,
            bootstrap=self.bootstrap
        )


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        # Merge all batches as batch processing is not possible
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
