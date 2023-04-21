import functools
import os
from typing import Tuple, Optional, Union, Callable, Dict, Any, List
import tempfile
import subprocess
import shutil
import logging

import numpy as np
import pandas as pd
import torch

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.data.preprocessing.exathlon import preprocess_exathlon_data
from timesead.utils.metadata import DATA_DIRECTORY

_logger = logging.getLogger(__name__)

TRAIN_FILES = (
    '1_0_1000000_14.csv',
    '1_0_100000_15.csv',
    '1_0_100000_16.csv',
    '1_0_10000_17.csv',
    '1_0_500000_18.csv',
    '1_0_500000_19.csv',
    '2_0_100000_20.csv',
    '2_0_100000_22.csv',
    '2_0_1200000_21.csv',
    '3_0_100000_24.csv',
    '3_0_100000_25.csv',
    '3_0_100000_26.csv',
    '3_0_1200000_23.csv',
    '4_0_1000000_31.csv',
    '4_0_100000_27.csv',
    '4_0_100000_28.csv',
    '4_0_100000_29.csv',
    '4_0_100000_30.csv',
    '4_0_100000_32.csv',
    '5_0_100000_33.csv',
    '5_0_100000_34.csv',
    '5_0_100000_35.csv',
    '5_0_100000_36.csv',
    '5_0_100000_37.csv',
    '5_0_100000_40.csv',
    '5_0_50000_38.csv',
    '5_0_50000_39.csv',
    '6_0_100000_42.csv',
    '6_0_100000_43.csv',
    '6_0_100000_44.csv',
    '6_0_100000_45.csv',
    '6_0_100000_46.csv',
    '6_0_100000_52.csv',
    '6_0_1200000_41.csv',
    '6_0_300000_50.csv',
    '6_0_50000_47.csv',
    '6_0_50000_48.csv',
    '6_0_50000_49.csv',
    '6_0_50000_51.csv',
    '9_0_100000_1.csv',
    '9_0_100000_3.csv',
    '9_0_100000_4.csv',
    '9_0_100000_6.csv',
    '9_0_1200000_2.csv',
    '9_0_300000_5.csv',
    '10_0_100000_10.csv',
    '10_0_100000_11.csv',
    '10_0_100000_13.csv',
    '10_0_100000_8.csv',
    '10_0_100000_9.csv',
    '10_0_1200000_7.csv',
    '10_0_300000_12.csv',
)


TEST_FILES = (
    '1_2_100000_68.csv',
    '1_4_1000000_80.csv',
    '1_5_1000000_86.csv',
    '2_1_100000_60.csv',
    '2_2_200000_69.csv',
    '2_5_1000000_87.csv',
    '2_5_1000000_88.csv',
    '3_2_1000000_71.csv',
    '3_2_500000_70.csv',
    '3_4_1000000_81.csv',
    '3_5_1000000_89.csv',
    '4_1_100000_61.csv',
    '4_5_1000000_90.csv',
    '5_1_100000_63.csv',
    '5_1_100000_64.csv',
    '5_1_500000_62.csv',
    '5_2_1000000_72.csv',
    '5_4_1000000_82.csv',
    '5_5_1000000_91.csv',
    '5_5_1000000_92.csv',
    '6_1_500000_65.csv',
    '6_3_200000_76.csv',
    '6_5_1000000_93.csv',
    '9_2_1000000_66.csv',
    '9_3_500000_74.csv',
    '9_4_1000000_78.csv',
    '9_5_1000000_84.csv',
    '10_2_1000000_67.csv',
    '10_3_1000000_75.csv',
    '10_4_1000000_79.csv',
    '10_5_1000000_85.csv',
)


TRAIN_LENGTHS = {
    1: [14391, 2690, 3591, 3591, 2728, 14391],
    2: [28725, 4269, 35923],
    3: [28790, 28790, 28789, 28791],
    4: [7191, 28789, 28769, 28790, 28790, 86391],
    5: [28790, 28790, 28791, 4742, 3590, 28790, 7191, 2727],
    6: [28757, 28789, 28790, 28790, 3588, 86390, 28790, 53990, 7190, 2634, 2690, 2689],
    9: [28790, 3345, 86341, 14390, 86391, 53990],
    10: [14356, 13224, 28790, 28746, 28790, 28790, 35989]
}


TEST_LENGTHS = {
    1: [2945, 43233, 3632],
    2: [46791, 2883, 43230, 3631],
    3: [2482, 2620, 4231, 5937],
    4: [129591, 3632],
    5: [43191, 46791, 46810, 2489, 4232, 43230, 3629],
    6: [46807, 46785, 3629],
    9: [7506, 46808, 43259, 5938],
    10: [10284, 46807, 43230, 5930],
}


class ExathlonDataset(BaseTSDataset):
    """
    Implements the Exathlon dataset from [Jacob2021]_.
    The data was collected by running different applications on a Spark cluster and recording metrics from the Spark
    service and the worker nodes. We consider the trace for each app a separate dataset. You can control which app trace
    to load by setting the `app_id` parameter.

    .. note::
        The Exathlon dataset consists of more than 2000 raw features that we reduce to 19 aggregated features as
        described in [Jacob2021]_. This is done in the preprocess step during the class initialization.

    .. note::
       Automatically downloading the dataset via the `download` option requires `git` to be installed on your system
       and is currently only tested on linux!

    .. warning::
        This dataset relies on preprocessing to be done on the data. Preprocessing can be done by setting the `preprocess`
        argument. The class will throw a RuntimeError without preprocessing.

    .. [Jacob2021] V. Jacob, F. Song, A. Stiegler, B. Rad, Y. Diao, and N. Tatbul.
        Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series.
        Proceedings of the VLDB Endowment (PVLDB), 14(11): 2613 - 2626, 2021.
    """
    GITHUB_LINK = 'https://github.com/exathlonbenchmark/exathlon.git'

    def __init__(self, dataset_path: str = os.path.join(DATA_DIRECTORY, 'exathlon'), app_id: int = 1,
                 training: bool = True, standardize: Union[bool, Callable[[pd.DataFrame, Dict], pd.DataFrame]] = True,
                 download: bool = True, preprocess: bool = True):
        """

        :param dataset_path: Folder from which to load the dataset.
        :param app_id: Data from which app to load. Must be in [1-6, 9, 10].
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        :param download: Whether to download the dataset if it doesn't exist.
        :param preprocess: Whether to setup the dataset for experiments.
        """

        if app_id not in TRAIN_LENGTHS:
            raise ValueError(f'App ID must be one of {list(TEST_LENGTHS.keys())}')

        self.dataset_path = dataset_path
        self.data_path = os.path.join(dataset_path, 'data', 'processed')
        self.app_id = app_id
        self.training = training

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')
        if not self._check_preprocessed():
            if not preprocess:
                raise RuntimeError('Dataset needs to be processed for proper working. Pass preprocess=True to setup the'
                                   ' dataset.')

            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")
            preprocess_exathlon_data(os.path.join(dataset_path, 'data', 'raw'), os.path.join(self.data_path))

        self.inputs = None
        self.targets = None

        if callable(standardize):
            with np.load(os.path.join(self.data_path, 'train', f'train_stats_{app_id}.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            with np.load(os.path.join(self.data_path, 'train', f'train_stats_{app_id}.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        test_str = 'train' if self.training else 'test'

        load_path = os.path.join(self.data_path, test_str)
        files = TRAIN_FILES if self.training else TEST_FILES
        files = [f for f in files if f.startswith(f'{self.app_id}_')]

        inputs, targets = [], []
        for f in files:
            file_name = os.path.join(load_path, f)

            data = pd.read_csv(file_name, index_col='t')

            if self.training:
                target = np.zeros(len(data), dtype=np.int64)
            else:

                target = data['Anomaly'].to_numpy()
                target = target != 0
                target = target.astype(np.int64)

            data = data.drop(columns=['Anomaly'])

            if self.standardize_fn is not None:
                data = self.standardize_fn(data)
            data = data.astype(np.float32)

            input = data.to_numpy()

            inputs.append(input)
            targets.append(target)

        return inputs, targets

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not (0 <= item < len(self)):
            raise KeyError('Out of bounds')

        if self.inputs is None or self.targets is None:
            self.inputs, self.targets = self.load_data()

        return (torch.as_tensor(self.inputs[item]),), (torch.as_tensor(self.targets[item]),)

    def __len__(self) -> Optional[int]:
        return len(TRAIN_LENGTHS[self.app_id]) if self.training else len(TEST_LENGTHS[self.app_id])

    @property
    def seq_len(self) -> List[int]:
        if self.training:
            return TRAIN_LENGTHS[self.app_id]
        else:
            return TEST_LENGTHS[self.app_id]

    @property
    def num_features(self) -> int:
        return 19

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {
            'subsample': {'class': 'SubsampleTransform', 'args': {'subsampling_factor': 5, 'aggregation': 'mean'}},
            'cache': {'class': 'CacheTransform', 'args': {}}
        }

    @staticmethod
    def get_feature_names():
        return ['driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value',
                'driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value',
                'driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value',
                '1_diff_driver_StreamingMetrics_streaming_totalCompletedBatches_value',
                '1_diff_driver_StreamingMetrics_streaming_totalProcessedRecords_value',
                '1_diff_driver_StreamingMetrics_streaming_totalReceivedRecords_value',
                '1_diff_driver_StreamingMetrics_streaming_lastReceivedBatch_records_value',
                '1_diff_driver_BlockManager_memory_memUsed_MB_value',
                '1_diff_driver_jvm_heap_used_value', '1_diff_node5_CPU_ALL_Idle%', '1_diff_node6_CPU_ALL_Idle%',
                '1_diff_node7_CPU_ALL_Idle%', '1_diff_node8_CPU_ALL_Idle%',
                '1_diff_avg_executor_filesystem_hdfs_write_ops_value', '1_diff_avg_executor_cpuTime_count',
                '1_diff_avg_executor_runTime_count', '1_diff_avg_executor_shuffleRecordsRead_count',
                '1_diff_avg_executor_shuffleRecordsWritten_count', '1_diff_avg_jvm_heap_used_value']

    def _check_exists(self) -> bool:
        # Only checks if the `data` folder exists
        data_folder_path = os.path.join(self.dataset_path, 'data')
        if not os.path.isdir(data_folder_path):
            return False
        return True

    def _check_preprocessed(self) -> bool:
        # Only checks if the `processed` folder exsits
        if not os.path.isdir(self.data_path):
            return False
        return True

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.dataset_path, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Checkout only the required parts from git https://stackoverflow.com/a/63786181/7196402
            _logger.info(f'Downloading Exathlon dataset from {self.GITHUB_LINK}...')
            subprocess.run(f'git clone --no-checkout --depth 1 {self.GITHUB_LINK}'.split(), cwd=temp_dir)
            git_dir = os.path.join(temp_dir, 'exathlon')
            subprocess.run('git sparse-checkout add data/ extract_data.sh'.split(), cwd=git_dir)
            subprocess.run('git checkout'.split(), cwd=git_dir)

            shutil.move(os.path.join(git_dir, 'data'), os.path.join(self.dataset_path, 'data'))
            shutil.move(os.path.join(git_dir, 'extract_data.sh'), os.path.join(self.dataset_path, 'extract_data.sh'))

        _logger.info('Extracting data from zip files...')
        ret = subprocess.run('extract_data.sh', cwd=self.dataset_path, shell=True)
        if not ret.returncode:
            _logger.error('Something went wrong during dataset download.')
        else:
            _logger.info('Done!')
