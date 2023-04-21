import logging
from collections import OrderedDict
from typing import List, Tuple, Optional, Union, Any, Dict

import numpy as np
import pandas as pd

import torch

import os

from timesead.data.dataset import BaseTSDataset
from timesead.data.transforms import OneVsRestTargetTransform
from timesead.utils.metadata import DATA_DIRECTORY
from timesead.data.preprocessing.tep import preprocess_tep_data

_logger = logging.getLogger(__name__)


class TEPDataset(BaseTSDataset):
    """
    Implementation of the Tennessee Eastman Process Dataset [Downs1993]_.
    The dataset was recorded by simulating a chemical process. The simulation also allows to introduce 20 different
    faults into the process which are used as anomaly labels. We implement the extended version of the dataset by
    Rieth et al. [Rieth2017]_ which runs the process several times with different RNG seeds.

    .. note::
       At the moment, we do not offer an automatic download option for this dataset. Please visit
       https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1 and download the files manually.

    .. warning::
        This dataset relies on preprocessing to be done on the data. Preprocessing can be done by setting the `preprocess`
        argument. The class will fail giving an error without preprocessing.

    .. [Downs1993] Downs, James J., and Ernest F. Vogel.
        "A plant-wide industrial process control problem." Computers & chemical engineering 17.3 (1993): 245-255.

    .. [Rieth2017] Rieth, Cory A.; Amsel, Ben D.; Tran, Randy; Cook, Maia B., 2017,
        "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation",
        https://doi.org/10.7910/DVN/6C3JR1, Harvard Dataverse, V1
    """
    def __init__(self, path: str = os.path.join(DATA_DIRECTORY, 'TEP_harvard'),
                 faults: Optional[Union[int, List[int]]] = None,
                 runs: Optional[Union[int, List[int]]] = None,
                 training: bool = True, standardize: bool = True, cache_size: int = 21, preprocess: bool = True):
        """

        :param path: Folder from which to load the dataset.
        :param faults: Specifies which faults to load data for. This can be a list of `int`\s, where 0 stands for
            fault-free data and [1, ..., 20] for the corresponding faults. Also supports a single `int` which means to
            only load data for this specific fault or `None` which loads data for all faults.
        :param runs: Specifies which runs to load for each fault. Each of the 500 runs was performed with a different
            random seed. This can either be specific runs passed as a list or a single `int` which means to load all runs
            from 0 up to this run. `None` means to load all available runs.
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        :param cache_size: Depending on the number of faults and runs chosen, this dataset can be quite large. It is
            therefore loaded in a lazy manner from disk. Data for each fault is kept in memory in a FIFO cache to reduce
            access time. This parameter sets the size of that cache. Setting this to the number of faults that you want
            to load will mean that eventually the entire dataset will be cached in memory.
        :param preprocess: Whether to setup dataset for experiments.
        """
        self.cache = OrderedDict()
        self.path = path
        self.processed_dir = os.path.join(self.path, 'processed')
        self.training = training
        self.standardize = standardize
        self.cache_size = cache_size

        if isinstance(faults, int):
            faults = [faults]
        elif faults is None:
            faults = list(range(0, 20 + 1))

        self.faults = set(faults)

        if not self._check_preprocessed():
            if not preprocess:
                raise RuntimeError('Dataset needs to be processed for proper working. Pass preprocess=True to setup the dataset.')

            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")
            os.makedirs(self.processed_dir, exist_ok=True)
            preprocess_tep_data(self.path, out_dir=self.processed_dir)

        if runs is None:
            # Select all runs
            runs = list(range(1, 500 + 1))
        elif isinstance(runs, int):
            runs = [runs]

        self.runs = runs

        if self.standardize:
            with np.load(os.path.join(self.processed_dir, 'TEP_FaultFree_Training_stats.npz')) as d:
                stats = dict(d)
            self.mean = stats['mean']
            self.std = stats['std']

    def load_data(self, fault: int, runs: Optional[Union[int, List[int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
        test_str = 'Training' if self.training else 'Testing'

        if runs is None:
            # Select all runs
            runs = list(range(1, 500 + 1))
        elif isinstance(runs, int):
            runs = [runs]

        # Check if data is in cache
        if (fault, self.training) in self.cache:
            # print(f'Cache HIT: fault {fault}, Training {self.training}')
            data = self.cache[(fault, self.training)]
        else:
            if fault == 0:
                fname = f'TEP_FaultFree_{test_str}.csv'
            else:
                fname = f'TEP_Faulty_{test_str}_{fault:02d}.csv'
            data = pd.read_csv(os.path.join(self.processed_dir, fname))

            # Set correct label for the first few samples, where the fault did not occur yet
            warmup = 20 if self.training else 160
            data.loc[data['sample'] <= warmup, 'faultNumber'] = 0

            if self.standardize:
                data[data.columns[3:]] -= self.mean
                data[data.columns[3:]] /= self.std
            data[data.columns[3:]] = data[data.columns[3:]].astype(np.float32)

            if len(self.cache) >= self.cache_size:
                # Drop if cache is full (FIFO)
                self.cache.popitem(last=False)

            # print(f'Cache MISS: fault {fault}, Training {self.training}')
            self.cache[(fault, self.training)] = data.copy()

        # Filter out unwanted runs
        data = data.loc[data['simulationRun'].isin(runs)]

        targets = data['faultNumber'].to_numpy()

        # Remove meta data
        data = data[data.columns[3:]]

        inputs = data.to_numpy()
        del data

        return inputs, targets

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not (0 <= item < len(self)):
            raise ValueError('Out of bounds')

        # Translate Index into specific run and fault number
        fault_index, run_index = divmod(item, len(self.runs))

        inputs, targets = self.load_data(fault_index, run_index+1)

        return (torch.as_tensor(inputs),), (torch.as_tensor(targets),)

    def __len__(self) -> Optional[int]:
        return len(self.faults) * len(self.runs)

    @property
    def seq_len(self) -> Optional[int]:
        return 500 if self.training else 960

    @property
    def num_features(self) -> int:
        return 52

    @staticmethod
    def get_feature_names() -> List[str]:
        return ['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9',
                'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17',
                'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25',
                'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32', 'xmeas_33',
                'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40', 'xmeas_41',
                'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11']

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {
            'onevsrest': {'class': OneVsRestTargetTransform, 'args': dict(normal_class=0, replace_labels=True)}
        }

    def _check_preprocessed(self):
        if not os.path.exists(os.path.join(self.processed_dir, 'TEP_FaultFree_Training_stats.npz')):
            return False

        test_str = 'Training' if self.training else 'Testing'
        for fault_no in self.faults:
            if fault_no == 0:
                file = f'TEP_FaultFree_{test_str}.csv'
            else:
                file = f'TEP_Faulty_{test_str}_{fault_no:02d}.csv'
            file_path = os.path.join(self.processed_dir, file)
            if not os.path.exists(file_path):
                return False
        return True
