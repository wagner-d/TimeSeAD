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
from .preprocessing import save_statistics

_logger = logging.getLogger(__name__)


class GenericDataset(BaseTSDataset):
    """
    Generic dataset class to load csv files as datasets.
    The data for training should be in a train/ folder and test data in a test/ folder.
    All csv files in the respective folders are used as part of the dataset.
    """

    def __init__(self, path: str, name: Optional[str]=None, separator: str=';',
                 features: Optional[List[str]]=None, anomaly_feature: Optional[str]=None,
                 training: bool=True, standardize: Union[bool, Callable]=True,
                 preprocess: bool=True, overwrite: bool=False):
        """
        :param path: Path to the dataset
        :param name: Name of the specific dataset file to be used.
            If not specified, all csv files are used.
        :param separator: Separator used in the dataset file
        :param features: List of features to be used in the dataset.
        :param anomaly_feature: Name of the feature that indicates anomalies.
        :param training: Whether to load the training or test data
        :param standardize: Can be either a boolean or a callable.
            If it is a callable, it is used to standardize the data.
            If it is a boolean, it indicates whether the data should be standardized.
        :param preprocess: Whether to setup the dataset for experiments.
        :param overwrite: Whether to overwrite existing files.
        """

        self.path = path
        self.processed_dir = os.path.join(self.path, 'processed')
        self.training = training
        self.standardize = standardize
        self.preprocess = preprocess
        self.separator = separator
        self.overwrite = overwrite
        self.features = features
        self.anomaly_feature = anomaly_feature

        test_str = 'train' if self.training else 'test'
        dataset_dir = os.path.join(self.path, test_str)
        if name is None:
            self.dataset_paths = glob.glob(os.path.join(dataset_dir, '*.csv'))
            self.stats_files = [os.path.join(self.processed_dir, f'{test_str}_{os.path.basename(f)}_stats.npz') for f in self.dataset_paths]
        else:
            self.dataset_paths = [os.path.join(dataset_dir, f'{name}.csv')]
            self.stats_files = [os.path.join(self.processed_dir, f'{test_str}_{name}_stats.npz')]

        self._setup_dataset()

        self.standardize_fns = []
        if callable(standardize):
            for stats_file in self.stats_files:
                with np.load(stats_file) as d:
                    stats = dict(d)
                standardize_fn = functools.partial(standardize, stats=stats)
                self.standardize_fns.append(standardize_fn)
        elif standardize:
            for stats_file in self.stats_files:
                with np.load(stats_file) as d:
                    stats = dict(d)
                standardize_fn = functools.partial(minmax_scaler, stats=stats)
                self.standardize_fns.append(standardize_fn)
        else:
            standardize_fn = [None] * len(self.stats_files)


    def _setup_dataset(self) -> None:
        self._seq_lens = []
        for index, stats_file in enumerate(self.stats_files):
            if os.path.exists(stats_file):
                data = self._read_dataset(index)
            else:
                data = self._generate_dataset_files(index)

            self._seq_lens.append(data.shape[0])
            # Final dataset features are used if not specified
            if not self.features:
                self.features = list(data.keys())
            self._num_features = len(self.features)


    def _read_dataset(self, dataset_index: int) -> pd.DataFrame:
        data = pd.read_csv(self.dataset_paths[dataset_index], sep=self.separator)
        return data


    def _generate_dataset_files(self, dataset_index: int) -> pd.DataFrame:
        os.makedirs(self.processed_dir, exist_ok=True)
        if not os.path.exists(self.dataset_paths[dataset_index]):
            raise FileNotFoundError(f"Dataset file {self.dataset_paths[dataset_index]} does not exist.")
        data = self._read_dataset(dataset_index)

        save_statistics(data, self.stats_files[dataset_index])

        # Drop columns with NaN values and notify the user
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            _logger.warning(f"Columns with NaN values: {nan_columns}. Dropping them.")
            data = data.dropna(axis=1, how='all')
            if not self.overwrite and not self.features:
                raise ValueError("NaN or non-float values found in dataset. Please preprocess the dataset first, mention which features to use, or set overwrite=True.")

        if self.overwrite:
            data.to_csv(self.dataset_paths[dataset_index], index=False, header=True, sep=self.separator)
        return data


    @functools.cache
    def load_data(self, dataset_index: int) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_csv(self.dataset_paths[dataset_index], sep=self.separator)
        if self.training:
            target = np.zeros(data.shape[0])
        else:
            target = data[self.anomaly_feature].to_numpy()

        if self.standardize_fns[dataset_index] is not None:
            data = self.standardize_fns[dataset_index](data)

        if self.features:
            data = data[self.features]

        return data.to_numpy(dtype=np.float32), target


    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not (0 <= item < len(self)):
            raise ValueError('Out of bounds')
        inputs, targets = self.load_data(item)

        return (torch.as_tensor(inputs),), (torch.as_tensor(targets),)


    def __len__(self) -> Optional[int]:
        return len(self._seq_lens)

    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self._seq_lens

    @property
    def num_features(self) -> int:
        return self._num_features

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {}

    @staticmethod
    def get_feature_names() -> List[str]:
        raise RuntimeError("Feature names are not available for generic datasets.")
