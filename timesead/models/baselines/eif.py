from typing import Optional, Tuple, Union

import torch
from numpy.lib.stride_tricks import sliding_window_view
from eif import iForest

from ..common import AnomalyDetector
from ...data.statistics import get_data_all

class EIFAD(AnomalyDetector):
    def __init__(
        self,
        n_trees: int = 200,
        sample_size: int = 256,
        extension_level: Optional[int] = None,
        window_size: int = 12,
        stride: int = 1,
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
            window_size[int]: Time frame window size to use for the model
            stride[int]: Stride value for sliding window
        """
        super(EIFAD, self).__init__()

        self.n_trees = n_trees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.window_size = window_size
        self.stride = stride
        self.input_shape = input_shape

    def _preprocess_data(self, input: torch.tensor) -> torch.tensor:
        # Converts data to window_size chunks
        flat_shape = (input.shape[0] - (self.window_size - 1), -1)
        slides = sliding_window_view(input, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
        return torch.from_numpy(slides)


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        data = self._preprocess_data(get_data_all(dataset.dataset))
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
        batch_input = inputs[0]
        # Convert input to (B, T, D) dimension
        if self.input_shape[0] == "t":
            batch_input = batch_input.permute(1, 0, 2)

        # Get the final window for each batch
        data = batch_input[:, -self.window_size:, :]
        # Reshape to match sliding_window_view output
        data = data.permute(0, 2, 1).reshape(data.shape[0], -1)

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
