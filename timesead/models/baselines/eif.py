from typing import Optional, Tuple, Union

import torch
from eif import iForest

from ..common import AnomalyDetector

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
            n_trees[int]: The number of trees in the forest.
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
        # Merge all batches as batch processing is not possible
        data_full = []
        for (b_inputs, b_targets) in dataset:
            data = b_inputs[0]
            batch_size, window_size, n_features = data.shape
            self.window_size = window_size
            data = data.reshape(batch_size, window_size*n_features)
            data_full.append(data)
        data_full = torch.cat(data_full)

        data = data_full.cpu().detach().numpy()
        extension_level = self.extension_level if self.extension_level != None else data.shape[1]-1
        self.model = iForest(data,
                             ntrees=self.n_trees,
                             sample_size=self.sample_size,
                             ExtensionLevel=extension_level)


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

        # iForest model doesn't seem to work with tensors
        data = data.cpu().detach().numpy()
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
