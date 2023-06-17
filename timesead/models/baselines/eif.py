from typing import Optional, Tuple, Union

import torch
import numpy as np
from eif import iForest

from ..common import AnomalyDetector
from ...data.statistics import get_data_all

class EIFAD(AnomalyDetector):
    def __init__(
        self,
        n_trees: int = 200,
        sample_size: int = 256,
        extension_level: Optional[int] = None,
        input_shape: str = "btf",
    ) -> None:
        """
        Extended Isolation Forest Anomaly Detector
            An implementation of the Extended Isolation Forest (EIF) for anomaly detection
            as described in [Hariri2019]_.

            Implementation derived from https://github.com/HPI-Information-Systems/TimeEval-algorithms

            .. [Hariri2019] S. Hariri, M. C. Kind and R. J. Brunner,
                "Extended Isolation Forest,"
                in IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 4, pp. 1479-1489,
                1 April 2021, doi: 10.1109/TKDE.2019.2947676.

        Args:
            sample_size[int]: The size of the subsample to be used in creation of each tree. Must be smaller than data size.
            extension_level[Optional[int]]: Specifies degree of freedom in choosing the hyperplanes for dividing up data. Must be smaller than the dimension n of the dataset.
                Value of 0 is identical to standard Isolation Forest, and None is equivalent to N-1 or fully extended
        """
        super(EIFAD, self).__init__()

        self.n_trees = n_trees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.input_shape = input_shape


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        data = get_data_all(dataset.dataset)
        # iForest model doesn't seem to work with tensors
        data = data.cpu().detach().numpy()
        extension_level = self.extension_level if self.extension_level != None else data.shape[1]-1
        self.model = iForest(data,
                             ntrees=self.n_trees,
                             sample_size=self.sample_size,
                             ExtensionLevel=extension_level)


    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        data = inputs[0]
        # Convert (T, B, D) to (B, T, D)
        if self.input_shape[0] == "t":
            data = data.permute(0, 2, 1)
        # iForest model doesn't seem to work with tensors
        data = data.cpu().detach().numpy()
        # Only the final point in window is needed for prediction
        data = data[:, -1, :]

        scores = torch.tensor(self.model.compute_paths(data))

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