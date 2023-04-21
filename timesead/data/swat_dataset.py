import functools
import logging
import os
from typing import Tuple, Optional, Union, Callable, Dict, Any

import numpy as np
import pandas as pd
import torch

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.data.preprocessing.swat import preprocess_swat_data
from timesead.utils.metadata import DATA_DIRECTORY

_logger = logging.getLogger(__name__)

class SWaTDataset(BaseTSDataset):
    """
    Implementation of the Secure WAter Treatment Dataset [Goh2016]_.
    This dataset was recorded from a miniature water treatment plant over the course of several weeks. Both training
    and test set consist of a single long time series, each. During testing, several attacks (cyber and physical) were
    carried out against the plant.

    .. note::
       Due to licensing issues, we cannot offer an automatic download option for this dataset. Please visit
       https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ and fill in the form to request a download link.
       The required files are in the folder `SWaT.A1 & A2_Dec 2015/Physical`.

    .. warning::
        This dataset relies on preprocessing to be done on the data. Preprocessing can be done by setting the `preprocess`
        argument to True. The class will fail giving an error without preprocessing.

    .. [Goh2016] Goh, Jonathan, et al. "A dataset to support research in the design of secure water treatment systems."
         Critical Information Infrastructures Security: 11th International Conference, CRITIS 2016, Paris, France,
         October 10–12, 2016, Revised Selected Papers 11. Springer International Publishing, 2017.
    """
    def __init__(self, path: str = os.path.join(DATA_DIRECTORY, 'SWaT', 'SWaT.A1 & A2_Dec 2015', 'Physical'),
                 training: bool = True, standardize: Union[bool, Callable] = True, remove_startup: bool = True, preprocess: bool = True):
        """

        :param path: Path where the files "SWaT_Dataset_Normal_v1.csv" and "SWaT_Dataset_Attack_v0.csv" are located.
        :param training: If True, this will load the training set consisting only of normal samples. Otherwise, loads
            the test set, which includes attacks.
        :param standardize: If True, apply min-max scaling (based on the training set). This can also be a function
            that accepts a DataFrame as its positional argument and a keyword argument `stats`: a dictionary of training
            data statistics.
        :param remove_startup: If True, this will remove the first 5 hours from the training set, as during this time
            the system was starting from an empty state. To be more exact, this removes only 4.5 hours, since the first 30
            minutes were already removed in v1 of the Dataset.
        :param preprocess: If True, setup dataset to run experiments.
        """
        self.path = path
        self.processed_dir = os.path.join(self.path, 'processed')
        self.training = training
        self.remove_startup = remove_startup

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Request and download the dataset from '
                               'https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/')
        if not self._check_preprocessed():
            if not preprocess:
                raise RuntimeError('Dataset needs to be processed for proper working. Pass preprocess=True to setup the'
                                   ' dataset.')

            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")
            preprocess_swat_data(self.path, out_dir=self.processed_dir)

        self.inputs = None
        self.targets = None

        if callable(standardize):
            with np.load(os.path.join(self.processed_dir, 'SWaT_Dataset_Normal_v1_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            with np.load(os.path.join(self.processed_dir, 'SWaT_Dataset_Normal_v1_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        test_str = 'Normal_v1' if self.training else 'Attack_v0'

        fname = f'SWaT_Dataset_{test_str}.csv'
        data = pd.read_csv(os.path.join(self.processed_dir, fname))

        # Convert string to int label
        data['Normal/Attack'] = (data['Normal/Attack'] == 'Attack').astype(np.int64)

        if self.standardize_fn is not None:
            data[data.columns[1:-1]] = self.standardize_fn(data[data.columns[1:-1]])
        data[data.columns[1:-1]] = data[data.columns[1:-1]].astype(np.float32)

        targets = data['Normal/Attack'].to_numpy()

        # Remove meta data
        data = data[data.columns[1:-1]]

        inputs = data.to_numpy()
        del data

        if self.training and self.remove_startup:
            inputs = inputs[12600:]
            targets = targets[12600:]

        return inputs, targets

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not (0 <= item < len(self)):
            raise ValueError('Out of bounds')

        if self.inputs is None or self.targets is None:
            self.inputs, self.targets = self.load_data()

        return (torch.as_tensor(self.inputs),), (torch.as_tensor(self.targets),)

    def __len__(self) -> Optional[int]:
        return 1

    @property
    def seq_len(self) -> Optional[int]:
        if self.training:
            if not self.remove_startup:
                return 495000
            else:
                return 482400
        else:
            return 449919

    @property
    def num_features(self) -> int:
        return 51

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {
            'subsample': {'class': 'SubsampleTransform', 'args': {'subsampling_factor': 5, 'aggregation': 'first'}},
            'cache': {'class': 'CacheTransform', 'args': {}}
        }

    @staticmethod
    def get_feature_names():
        return ['FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203','FIT201','MV201','P201','P202',
                'P203','P204','P205','P206','DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302',
                'AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503',
                'AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503','FIT601','P601',
                'P602','P603']

    def _check_exists(self) -> bool:
        if not os.path.exists(self.path):
            return False
        return True

    def _check_preprocessed(self) -> bool:
        exists = os.path.exists(os.path.join(self.processed_dir, 'SWaT_Dataset_Normal_v1_stats.npz')) and \
                 os.path.exists(os.path.join(self.processed_dir, 'SWaT_Dataset_Normal_v1.csv')) and \
                 os.path.exists(os.path.join(self.processed_dir, 'SWaT_Dataset_Attack_v0.csv'))

        return exists
