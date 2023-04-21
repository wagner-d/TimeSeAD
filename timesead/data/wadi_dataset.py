import functools
import logging
import os
from typing import Tuple, Optional, Union, Callable, Dict, Any

import numpy as np
import pandas as pd
import torch

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.data.preprocessing.wadi import preprocess_wadi_data
from timesead.utils.metadata import DATA_DIRECTORY

_logger = logging.getLogger(__name__)


class WADIDataset(BaseTSDataset):
    """
    Implementation of the WAter DIstribution Dataset [Ahmed2017]_.
    This dataset was recorded from a miniature water distribution network over the course of several weeks.
    Both training and test set consist of a single long time series, or two time series, see details about the `split`
    parameter. During testing, several attacks (cyber and physical) were carried out against the plant.

    .. note::
       Due to licensing issues, we cannot offer an automatic download option for this dataset. Please visit
       https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ and fill in the form to request a download link.
       The required files are in the folder `WADI.A2_19 Nov 2019`.

    .. [Ahmed2017] Ahmed, Chuadhry Mujeeb, Venkata Reddy Palleti, and Aditya P. Mathur.
        "WADI: a water distribution testbed for research in the design of secure cyber physical systems."
        Proceedings of the 3rd international workshop on cyber-physical systems for smart water networks. 2017.
    """
    def __init__(self, path: str = os.path.join(DATA_DIRECTORY, 'wadi', 'WADI.A2_19 Nov 2019'),
                 training: bool = True, standardize: Union[bool, Callable[[pd.DataFrame, Dict], pd.DataFrame]] = True,
                 remove_startup: bool = True, split: bool = True, preprocess: bool = True):
        """

        :param path: Folder from which to load the dataset.
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        :param remove_startup: This removes the first 5 hours of the training set, during which the plant is starting.
        :param split: The authors removed some data points in v2 of the training dataset. Thus, there is a clear split
            at index 335998. Setting this to true will return 2 TS split at this location. Otherwise, one long TS is
            returned.
        :param preprocess: Whether to setup the dataset for experiments.
        """
        self.path = path
        self.processed_dir = os.path.join(self.path, 'processed')
        self.training = training
        self.remove_startup = remove_startup
        self.split = split

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Request and download the dataset from https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/')
        if preprocess and not self._check_preprocessed():
            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")
            os.makedirs(self.processed_dir, exist_ok=True)
            preprocess_wadi_data(self.path, self.processed_dir)

        self.startup_remove_amount = 18000  # 5h
        self.split_index = 335999

        self.inputs = None
        self.targets = None

        if callable(standardize):
            with np.load(os.path.join(self.processed_dir, 'WADI_14days_new_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            with np.load(os.path.join(self.processed_dir, 'WADI_14days_new_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        test_str = 'WADI_14days_new' if self.training else 'WADI_attackdataLABLE'

        fname = f'{test_str}.csv'
        skiprows = 0 if self.training else 1
        data = pd.read_csv(os.path.join(self.path, fname), skiprows=skiprows)

        if self.training:
            targets = np.zeros(len(data), dtype=np.int64)
            # Remove columns for row index, date, and time and sensors that are missing for every time step
            data = data.drop(columns=['Row', 'Date', 'Time', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS',
                                      '2_P_002_STATUS'])
            nan_rows, = np.nonzero(data.isna().any(axis=1).to_numpy())
            for i in nan_rows:
                row = data.iloc[i]
                nonzero, = np.nonzero(row.isna().to_numpy())
                data.iloc[i, nonzero] = data.iloc[i - 1, nonzero]
        else:
            # Remove last two rows as they are empty
            data = data.iloc[:-2]
            # Convert 1/-1 labels to 0/1 labels
            targets = -data['Attack LABLE (1:No Attack, -1:Attack)'].to_numpy() * 0.5 + 0.5
            targets = targets.astype(np.int64)
            data = data.drop(columns=['Attack LABLE (1:No Attack, -1:Attack)'])
            # Remove columns for row index, date, and time and columns that contain only nan values
            data = data.drop(columns=['Row ', 'Date ', 'Time', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS',
                                      '2_P_002_STATUS'])

        if self.standardize_fn is not None:
            data = self.standardize_fn(data)
        data = data.astype(np.float32)

        inputs = data.to_numpy()

        if self.training and self.remove_startup:
            inputs = inputs[self.startup_remove_amount:]
            targets = targets[self.startup_remove_amount:]

        if self.training and self.split:
            split_index = self.split_index - self.startup_remove_amount if self.remove_startup else self.split_index
            inputs = np.array_split(inputs, [split_index], axis=0)
            targets = np.array_split(targets, [split_index], axis=0)

        return inputs, targets

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """
        This should return the time series of the dataset. I.e., if the dataset has 5 independent time-series,
        passing 0, ..., 4 as item should return these time series. The format is (inputs, targets), where inputs
        and targets are tupples of torch.Tensors.

        :param item: Index of the time series to return.
        :return:
        """
        if not (0 <= item < len(self)):
            raise KeyError('Out of bounds')

        if self.inputs is None or self.targets is None:
            self.inputs, self.targets = self.load_data()

        if self.split and self.training:
            return (torch.as_tensor(self.inputs[item]),), (torch.as_tensor(self.targets[item]),)

        return (torch.as_tensor(self.inputs),), (torch.as_tensor(self.targets),)

    def __len__(self) -> Optional[int]:
        return 2 if self.split and self.training else 1

    @property
    def seq_len(self) -> Optional[int]:
        if self.training:
            split1 = self.split_index - self.startup_remove_amount if self.remove_startup else self.split_index
            split2 = 784571 - self.split_index
            return [split1, split2] if self.split else split1 + split2
        else:
            return 172801

    @property
    def num_features(self) -> int:
        return 123

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {
            'subsample': {'class': 'SubsampleTransform', 'args': {'subsampling_factor': 5, 'aggregation': 'first'}},
            'cache': {'class': 'CacheTransform', 'args': {}}
        }

    @staticmethod
    def get_feature_names():
        return ['1_AIT_001_PV','1_AIT_002_PV','1_AIT_003_PV','1_AIT_004_PV','1_AIT_005_PV','1_FIT_001_PV','1_LS_001_AL',
                '1_LS_002_AL','1_LT_001_PV','1_MV_001_STATUS','1_MV_002_STATUS','1_MV_003_STATUS','1_MV_004_STATUS',
                '1_P_001_STATUS','1_P_002_STATUS','1_P_003_STATUS','1_P_004_STATUS','1_P_005_STATUS','1_P_006_STATUS',
                '2_DPIT_001_PV','2_FIC_101_CO','2_FIC_101_PV','2_FIC_101_SP','2_FIC_201_CO','2_FIC_201_PV',
                '2_FIC_201_SP','2_FIC_301_CO','2_FIC_301_PV','2_FIC_301_SP','2_FIC_401_CO','2_FIC_401_PV',
                '2_FIC_401_SP','2_FIC_501_CO','2_FIC_501_PV','2_FIC_501_SP','2_FIC_601_CO','2_FIC_601_PV',
                '2_FIC_601_SP','2_FIT_001_PV','2_FIT_002_PV','2_FIT_003_PV','2_FQ_101_PV','2_FQ_201_PV','2_FQ_301_PV',
                '2_FQ_401_PV','2_FQ_501_PV','2_FQ_601_PV','2_LS_101_AH','2_LS_101_AL',
                '2_LS_201_AH','2_LS_201_AL','2_LS_301_AH','2_LS_301_AL','2_LS_401_AH','2_LS_401_AL','2_LS_501_AH',
                '2_LS_501_AL','2_LS_601_AH','2_LS_601_AL','2_LT_001_PV','2_LT_002_PV','2_MCV_007_CO','2_MCV_101_CO',
                '2_MCV_201_CO','2_MCV_301_CO','2_MCV_401_CO','2_MCV_501_CO','2_MCV_601_CO','2_MV_001_STATUS',
                '2_MV_002_STATUS','2_MV_003_STATUS','2_MV_004_STATUS','2_MV_005_STATUS','2_MV_006_STATUS',
                '2_MV_009_STATUS','2_MV_101_STATUS','2_MV_201_STATUS','2_MV_301_STATUS','2_MV_401_STATUS',
                '2_MV_501_STATUS','2_MV_601_STATUS','2_P_003_SPEED','2_P_003_STATUS',
                '2_P_004_SPEED','2_P_004_STATUS','2_PIC_003_CO','2_PIC_003_PV','2_PIC_003_SP','2_PIT_001_PV',
                '2_PIT_002_PV','2_PIT_003_PV','2_SV_101_STATUS','2_SV_201_STATUS','2_SV_301_STATUS','2_SV_401_STATUS',
                '2_SV_501_STATUS','2_SV_601_STATUS','2A_AIT_001_PV','2A_AIT_002_PV','2A_AIT_003_PV','2A_AIT_004_PV',
                '2B_AIT_001_PV','2B_AIT_002_PV','2B_AIT_003_PV','2B_AIT_004_PV','3_AIT_001_PV','3_AIT_002_PV',
                '3_AIT_003_PV','3_AIT_004_PV','3_AIT_005_PV','3_FIT_001_PV','3_LS_001_AL','3_LT_001_PV',
                '3_MV_001_STATUS','3_MV_002_STATUS','3_MV_003_STATUS','3_P_001_STATUS','3_P_002_STATUS',
                '3_P_003_STATUS','3_P_004_STATUS','LEAK_DIFF_PRESSURE','PLANT_START_STOP_LOG','TOTAL_CONS_REQUIRED_FLOW']

    def _check_exists(self) -> bool:
        if not os.path.exists(self.path):
            return False
        return True

    def _check_preprocessed(self) -> bool:
        stats_file = os.path.join(self.processed_dir, 'WADI_14days_new_stats.npz')
        if os.path.exists(stats_file):
            return True
        return False
