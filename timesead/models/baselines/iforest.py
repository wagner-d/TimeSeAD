from typing import Optional, Tuple, Union

import torch
import numpy
from pyod.models.iforest import IForest

from ..common import AnomalyDetector
from ...data.statistics import get_data_all

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