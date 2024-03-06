from typing import List, Optional, Union, Callable, Tuple, Dict, Any
import os
import logging
import glob
import pandas as pd
import numpy as np
import functools
import torch

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.utils.metadata import DATA_DIRECTORY
from .preprocessing import save_statistics

_logger = logging.getLogger(__name__)


class GenericDataset(BaseTSDataset):
    """

    """

    def __init__(self, path: str, name: Optional[str]=None, separator: str=';',
                 training: bool=True, standardize: Union[bool, Callable]=True,
                 preprocess: bool=True):
        """
        :param path: Path to the dataset
        :param name: Name of the specific dataset file to be used.
            If not specified, the dataset files are combined.
        :param training: Whether to load the training or test data
        :param standardize: Can be either a boolean or a callable.
            If it is a callable, it is used to standardize the data.
            If it is a boolean, it indicates whether the data should be standardized.
        :param preprocess: Whether to setup the dataset for experiments.
        """

        self.path = path
        self.processed_dir = os.path.join(self.path, 'processed')
        self.training = training
        self.standardize = standardize
        self.preprocess = preprocess

        dataset_name = name if name is not None else 'combined'
        test_str = 'train' if self.training else 'test'
        self.dataset_path = os.path.join(self.processed_dir, f'{test_str}_{dataset_name}.csv')
        self.stats_file = os.path.join(self.processed_dir, f'{test_str}_{dataset_name}_stats.npz')

        # TODO(AR): have an additional file that indicates which files were combined
        self._setup_dataset(name, separator)

        self.inputs  = None
        self.targets = None

        if callable(standardize):
            with np.load(self.stats_file) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            with np.load(self.stats_file) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None


    def _setup_dataset(self, name: str, separator: str) -> None:
        if os.path.exists(self.dataset_path) and os.path.exists(self.stats_file):
            data = np.genfromtxt(self.dataset_path, delimiter=',', dtype=np.float32)
            data = pd.DataFrame(data)
        else:
            data = self._generate_dataset_files(name, separator)

        self._seq_len = data.shape[0]
        self.features = list(data.items())
        self._num_features = len(self.features)


    def _generate_dataset_files(self, name: Optional[str], separator: str) -> pd.DataFrame:
        os.makedirs(self.processed_dir, exist_ok=True)
        data = None
        if name is not None:
            raise NotImplementedError("Loading a specific dataset file is not implemented yet.")
        else:
            test_str = 'train' if self.training else 'test'
            for csv_file in glob.glob(os.path.join(self.path, test_str, '*.csv')):
                part_data = pd.read_csv(csv_file, sep=separator)
                if data is None:
                    data = part_data
                else:
                    data = pd.concat([data, part_data])

        # Drop columns with NaN values and notify the user
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            _logger.warning(f"Columns with NaN values: {nan_columns}. Dropping them.")
            data = data.dropna(axis=1, how='all')

        data.to_csv(self.dataset_path, index=False, header=False, sep=',')
        save_statistics(data, self.stats_file)
        return data


    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.genfromtxt(self.dataset_path, delimiter=',', dtype=np.float32)
        if self.training:
            target = np.zeros(data.shape[0])
        else:
            raise NotImplementedError("Test data not implemented yet.")

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
        return self._seq_len

    @property
    def num_features(self) -> int:
        return self._num_features

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {}

    @staticmethod
    def get_feature_names() -> List[str]:
        raise RuntimeError("Feature names are not available for generic datasets.")
