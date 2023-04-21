import logging
from typing import Tuple, Optional, Dict, Any, Union, List, Callable
import os

import torch.utils.data
import numpy as np
import functools

from timesead.data.preprocessing.smd import preprocess_smd_data

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.utils.metadata import DATA_DIRECTORY

_logger = logging.getLogger(__name__)

FILENAMES = [
    'machine-1-3.txt',
    'machine-1-7.txt',
]

TRAIN_LENS = [500, 1000]

TEST_LENS = [500, 1000]


class MiniSMDDataset(BaseTSDataset):
    """
    This is a condensed version of the :class:`~timesead.data.smd_dataset.SMDDataset` containing only shortened time
    series for two different machines. Mostly used for testing purposes.
    """
    def __init__(self, server_id: int = 0, path: str = os.path.join(DATA_DIRECTORY, 'mini_smd'),
                 training: bool = True, standardize: Union[bool, Callable] = True, preprocess: bool = True):
        """

        :param server_id: ID of the server to load. Must be 0 or 1.
        :param path: Path to the data
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        """
        self.server_id   = server_id
        self.path        = path
        self.training    = training
        self.standardize = standardize

        self.inputs  = None
        self.targets = None

        self.processed_dir = os.path.join(self.path, 'processed')

        if preprocess and not self._check_preprocessed():
            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")
            os.makedirs(self.processed_dir, exist_ok=True)
            preprocess_smd_data(self.path, self.processed_dir, FILENAMES)

        if callable(standardize):
            with np.load(os.path.join(self.processed_dir, FILENAMES[self.server_id].split('.')[0] + '_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            with np.load(os.path.join(self.processed_dir, FILENAMES[self.server_id].split('.')[0] + '_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:

        test_str = 'train' if self.training else 'test'

        filename = FILENAMES[self.server_id]

        data = np.genfromtxt(os.path.join(self.path, test_str, filename), dtype=np.float32, delimiter=',')

        if self.training:
            target = np.zeros(data.shape[0])
        else:
            target = np.genfromtxt(os.path.join(self.path, 'test_label', filename), dtype=np.float32, delimiter=',')

        if self.standardize_fn is not None:
            data = self.standardize_fn(data)

        return data, target

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not (0 <= item < len(self)):
            raise ValueError('Out of bounds')

        if self.inputs is None or self.targets is None:
            self.inputs, self.targets = self.load_data()

        return (torch.as_tensor(self.inputs),), (torch.as_tensor(self.targets),)

    def __len__(self) -> Optional[int]:
        return 1

    @property
    def seq_len(self) -> Union[int, List[int]]:
        if self.training:
            return TRAIN_LENS[self.server_id]
        else:
            return TEST_LENS[self.server_id]

    @property
    def num_features(self) -> int:
        return 38

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {}

    @staticmethod
    def get_feature_names():
        return [''] * 38

    def _check_preprocessed(self) -> bool:
        stats_file = os.path.join(self.processed_dir, FILENAMES[self.server_id].split('.')[0] + '_stats.npz')
        return  os.path.exists(stats_file)
