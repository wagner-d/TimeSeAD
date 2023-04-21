from typing import Tuple, Optional, Dict, Any, Union, List, Callable
import os

import torch.utils.data
import numpy as np
import functools
import tempfile
import logging
import subprocess
import shutil

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.data.preprocessing.smd import preprocess_smd_data
from timesead.utils.metadata import DATA_DIRECTORY

_logger = logging.getLogger(__name__)

FILENAMES = [
    'machine-1-1.txt',
    'machine-1-2.txt',
    'machine-1-3.txt',
    'machine-1-4.txt',
    'machine-1-5.txt',
    'machine-1-6.txt',
    'machine-1-7.txt',
    'machine-1-8.txt',
    'machine-2-1.txt',
    'machine-2-2.txt',
    'machine-2-3.txt',
    'machine-2-4.txt',
    'machine-2-5.txt',
    'machine-2-6.txt',
    'machine-2-7.txt',
    'machine-2-8.txt',
    'machine-2-9.txt',
    'machine-3-1.txt',
    'machine-3-10.txt',
    'machine-3-11.txt',
    'machine-3-2.txt',
    'machine-3-3.txt',
    'machine-3-4.txt',
    'machine-3-5.txt',
    'machine-3-6.txt',
    'machine-3-7.txt',
    'machine-3-8.txt',
    'machine-3-9.txt'
]

TRAIN_LENS = [28479, 23694, 23702, 23706, 23705, 23688, 23697, 23698, 23693, 23699, 23688, 23689, 23688, 28743, 23696,
              23702, 28722, 28700, 23692, 28695, 23702, 23703, 23687, 23690, 28726, 28705, 28703, 28713]

TEST_LENS = [28479, 23694, 23703, 23707, 23706, 23689, 23697, 23699, 23694, 23700, 23689, 23689, 23689, 28743, 23696,
             23703, 28722, 28700, 23693, 28696, 23703, 23703, 23687, 23691, 28726, 28705, 28704, 28713]


class SMDDataset(BaseTSDataset):
    """
    Implementation of the Server Machine Dataset [Su2019]_.
    The data consists of traces from 28 different servers recorded over several weeks. We consider each trace to be a
    separate dataset.

    .. note::
        Automatically downloading the dataset currently requires that you have `git` installed on your system!

    .. [Su2019] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, D. Pei.
        Robust anomaly detection for multivariate time series through stochastic recurrent neural network.
        In: Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining,
        2019 Jul 25 (pp. 2828-2837).
    """
    GITHUB_LINK = 'https://github.com/NetManAIOps/OmniAnomaly.git'

    def __init__(self, server_id: int, path: str = os.path.join(DATA_DIRECTORY, 'smd'),
                 training: bool = True, standardize : Union[bool, Callable] = True,
                 download: bool = True, preprocess: bool = True):
        """

        :param path: Folder from which to load the dataset.
        :param server_id: Data from which machine to load. Must be in [0, ..., 27].
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        :param download: Whether to download the dataset if it doesn't exist.
        :param preprocess: Whether to setup the dataset for experiments.
        """
        if not (0 <= server_id <= 27):
            raise ValueError(f'Server ID must be between 0 and 27! Given: {server_id}')

        self.server_id   = server_id
        self.path        = path
        self.processed_dir = os.path.join(self.path, 'processed')
        self.training    = training
        self.standardize = standardize
        stats_file = os.path.join(self.processed_dir, FILENAMES[server_id].split('.')[0] + '_stats.npz')

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')
        if preprocess and not self._check_preprocessed():
            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")

            os.makedirs(self.processed_dir, exist_ok=True)
            preprocess_smd_data(self.path, self.processed_dir, FILENAMES)

        self.inputs  = None
        self.targets = None

        if callable(standardize):
            with np.load(stats_file) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            with np.load(stats_file) as d:
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

    def _check_exists(self) -> bool:
        # Checks if the relevant folders exist but not their content
        folder_paths = [os.path.join(self.path, folder)
                        for folder in ['interpretation_label', 'test', 'test_label', 'train']]
        if not all(os.path.isdir(folder_path) for folder_path in folder_paths):
            return False
        return True

    def _check_preprocessed(self) -> bool:
        stats_file = os.path.join(self.processed_dir, FILENAMES[self.server_id].split('.')[0] + '_stats.npz')
        if os.path.exists(stats_file):
            return True
        return False

    def download(self):
        if self._check_exists():
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download only the required parts from git https://stackoverflow.com/a/63786181/7196402
            _logger.info(f'Downloading SMD dataset from {self.GITHUB_LINK}...')
            subprocess.run(f'git clone --no-checkout --depth 1 {self.GITHUB_LINK}'.split(), cwd=temp_dir)
            git_dir = os.path.join(temp_dir, 'OmniAnomaly')
            subprocess.run('git sparse-checkout add ServerMachineDataset/'.split(), cwd=git_dir)
            subprocess.run('git checkout'.split(), cwd=git_dir)

            shutil.move(os.path.join(git_dir, 'ServerMachineDataset'), self.path)

        _logger.info('Done!')
