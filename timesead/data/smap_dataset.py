import abc
import ast
import csv
import errno
import functools
import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from typing import Tuple, List, Union, Dict, Any

import numpy as np
import torch

from timesead.data.dataset import BaseTSDataset
from timesead.utils.metadata import DATA_DIRECTORY, PROJECT_ROOT
from timesead.utils.utils import getitem

DATASET_URL = r'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
LABELS_URL = r'https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv'
BUFFER_SIZE = 16*1024*1024
ZIP_CHECKSUM = 'b4d66deb492d9b0a353b51879152687ed9313897e8e19320d2dc853d738ed8a7'
FILE_CHECKSUMS = os.path.join(PROJECT_ROOT, 'data', 'smap', 'smap_checksums.json')


class SMAPDownloader:
    """
    Class that downloads and extracts the SMAP and MSL datasets [Hundman2018]_.
    Files are also checked for integrity against their SHA-256 hashes stored in data/SMAP/smap_checksums.json.

    .. [Hundman2018] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom.
        Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding.
        In: Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining,
        2018 Jul 19 (pp. 387-395).
    """
    def __init__(self, data_path: str = os.path.join(DATA_DIRECTORY, 'smap')):
        """

        :param data_path: The folder in which to download the dataset.
        """
        self.data_path = data_path

    @staticmethod
    def compute_sha256(file, buffer_size: int = BUFFER_SIZE) -> str:
        """
        Compute the SHA-256 hash of a file object.

        :param file: A file object returned by :func:`open`. Note that it should be opened in binary mode.
        :param buffer_size: Data from the file is read in chunks. This specifies the chunk size in bytes.
        :return: The SHA-256 hash of the file as a hex string.
        """
        hasher = hashlib.sha256()
        while True:
            data = file.read(buffer_size)
            if len(data) == 0:
                break
            hasher.update(data)

        return hasher.hexdigest()

    def check_existing_files(self) -> bool:
        """
        Checks if all files specified in the `FILE_CHECKSUMS` json file are present and if their checksums are correct.

        :return: `True` if all files are present and their checksums are correct, `False` otherwise.
        """
        if not os.path.isdir(self.data_path):
            return False

        # Check checksums for all relevant files
        with open(FILE_CHECKSUMS, mode='r') as f:
            checksums = json.load(f)
        for file, chksum in checksums.items():
            file_path = os.path.join(self.data_path, file)
            if not os.path.isfile(file_path):
                return False

            with open(file_path, mode='rb') as f:
                file_hash = self.compute_sha256(f)
            if file_hash != chksum:
                logging.error(f'SHA-256 checksum of file {file} is not correct! Expected "{chksum}", got "{file_hash}".')
                return False

        return True

    @staticmethod
    def download_to_file(url: str, file, buffer_size: int = BUFFER_SIZE):
        """
        Download a file from any URL supported by `urllib` to a file object.

        :param url: The URL of the file to download.
        :param file: Open file object to which the data is saved. This should be in binary mode.
        :param buffer_size: This method downloads data in chunks. `buffer_size` specifies the chunk size.
        """
        logging.info(f'Downloading "{url}"...')
        with urllib.request.urlopen(url) as data:
            shutil.copyfileobj(data, file, buffer_size)
        file.seek(0)

    def download_data(self):
        """
        Download the SMAP and MSL datasets.
        """
        if self.check_existing_files():
            return

        # download files
        try:
            os.makedirs(self.data_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        with tempfile.TemporaryFile() as tmp:
            self.download_to_file(DATASET_URL, tmp)

            if self.compute_sha256(tmp) != ZIP_CHECKSUM:
                raise RuntimeError('The SHA-256 Hash of the downloaded zip is not correct!')

            logging.info('Extracting data...')
            with zipfile.ZipFile(tmp, 'r') as zip:
                for info in zip.infolist():
                    if info.filename.startswith('data/test') or info.filename.startswith('data/train'):
                        info.filename = info.filename.replace('data/', '')
                        zip.extract(info, self.data_path)
            logging.info('Done!')

        with open(os.path.join(self.data_path, 'labeled_anomalies.csv'), mode='wb') as f:
            self.download_to_file(LABELS_URL, f)

        logging.info('Checking SHA-256 checksum of downloaded files...')
        if not self.check_existing_files():
            logging.critical('FAILURE!')
            raise RuntimeError('The SHA-256 Hash of the downloaded files is not correct!')


class _SMAPBaseDataset(BaseTSDataset, abc.ABC):
    def __init__(self, name: str, channels: List[str], train_lens: List[int], test_lens: List[int],
                 data_path: str = os.path.join(DATA_DIRECTORY, 'smap'), channel_id: int = 0, training: bool = True,
                 download: bool = True):
        super(_SMAPBaseDataset, self).__init__()
        self.data_path = data_path
        self.training = training
        self.channel_id = channel_id
        self.downloader = SMAPDownloader()
        self.name = name
        self.channels = channels
        self.train_lens = train_lens
        self.test_lens = test_lens

        if download:
            self.downloader.download_data()

        self.data = self.labels = None

    def load_data(self) -> Tuple[List[np.ndarray], ...]:
        with open(os.path.join(self.data_path, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=functools.partial(getitem, item=0))
        # Note: P-2 is excluded in the OmniAnomaly code for some reason
        data_info = [row for row in res if row[1] == self.name and row[0] == self.channels[self.channel_id]]

        labels = []
        if not self.training:
            for row in data_info:
                anomalies = ast.literal_eval(row[2])
                length = int(row[-1])
                label = np.zeros([length], dtype=np.int64)
                for anomaly in anomalies:
                    label[anomaly[0]:anomaly[1] + 1] = 1
                labels.append(label)

        def load_sequences(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(self.data_path, category, filename + '.npy'))
                data.append(temp.astype(np.float32))
                if self.training:
                    labels.append(np.zeros((temp.shape[0],), dtype=np.int64))
            return data

        return load_sequences('train' if self.training else 'test'), labels

    def __len__(self) -> int:
        return 1

    @property
    def seq_len(self) -> Union[int, List[int]]:
        if self.training:
            return self.train_lens[self.channel_id]

        return self.test_lens[self.channel_id]

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if self.data is None:
            self.data, self.labels = self.load_data()

        return (torch.as_tensor(self.data[item]),), (torch.as_tensor(self.labels[item]),)

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {}

    def get_feature_names(self) -> List[str]:
        # Feature names of binary features not really specified
        return [self.channels[self.channel_id]] + [''] * (self.num_features - 1)


class SMAPDataset(_SMAPBaseDataset):
    """
    Implementation of the SMAP dataset [Hundman2018].
    It consists of several monitored values from a single satellite and commands sent to that satellite. We consider the
    trace for each channel a separate dataset, where the monitored value is in the first feature dimension and the
    remaining binary features correspond to the commands.
    """
    def __init__(self, data_path: str = os.path.join(DATA_DIRECTORY, 'smap'), channel_id: int = 0,
                 training: bool = True, download: bool = True):
        """

        :param data_path: Folder from which to load the dataset.
        :param channel_id: Data from which channel to load. Must be in [0-54].
        :param training: Whether to load the training or the test set.
        :param download: Whether to download the dataset if it doesn't exist.
        """
        smap_channels = ['P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'E-10', 'E-11',
                         'E-12', 'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2', 'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1',
                         'G-2', 'D-5', 'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9', 'F-2', 'G-4',
                         'T-3', 'D-11', 'D-12', 'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 'P-2',
                         'A-8', 'A-9', 'F-3']

        smap_train_lens = [2872, 2818, 2880, 2880, 2880, 2880, 2880, 2880, 2872, 2818, 2880, 2880, 2880, 2880, 2880,
                           2880, 2769, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2849, 2821, 2855, 2880, 2880, 2833,
                           2648, 2736, 2690, 2820, 2478, 2561, 2594, 2583, 2869, 2609, 2624, 2875, 2855, 2602, 2583,
                           2861, 2551, 2876, 2611, 312, 2435, 2881, 2446, 2853, 2874, 705, 682, 2879, 1490, 2821, 762,
                           762, 2880]

        smap_test_lens = [8505, 7331, 8516, 8532, 8307, 8354, 8294, 8300, 8310, 8532, 8302, 8505, 8514, 8512, 8640,
                          8640, 8509, 8209, 8493, 8595, 8640, 8473, 7914, 8205, 8080, 8469, 7361, 7628, 7884, 7642,
                          8584, 7783, 7907, 8612, 8625, 7874, 7406, 8626, 7632, 8579, 7431, 7918, 8044, 8640, 8029,
                          8071, 7244, 4693, 4453, 8631, 7663, 8209, 8375, 8434, 8376]

        super(SMAPDataset, self).__init__('SMAP', smap_channels, smap_train_lens, smap_test_lens,
                                         data_path=data_path, channel_id=channel_id, training=training,
                                         download=download)

    @property
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return 25


class MSLDataset(_SMAPBaseDataset):
    """
    Implementation of the MSL dataset [Hundman2018].
    It consists of several monitored values from a mars rover and commands sent to the rover. We consider the trace for
    each channel a separate dataset, where the monitored value is in the first feature dimension and the remaining
    binary features correspond to the commands.
    """
    def __init__(self, data_path: str = os.path.join(DATA_DIRECTORY, 'smap'), channel_id: int = 0,
                 training: bool = True, download: bool = True):
        """

        :param data_path: Folder from which to load the dataset.
        :param channel_id: Data from which channel to load. Must be in [0-26].
        :param training: Whether to load the training or the test set.
        :param download: Whether to download the dataset if it doesn't exist.
        """
        msl_channels = ['M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4', 'M-5', 'P-15', 'C-1',
                        'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16',
                        'M-7', 'F-8']
        msl_train_lens = [1565, 2209, 2208, 926, 4308, 2272, 2272, 2511, 2037, 2076, 2032, 3682, 2158, 764, 1145, 1145,
                          2244, 2598, 3675, 439, 2880, 748, 3969, 2074, 1451, 1587, 3342]
        msl_test_lens = [2049, 2277, 2277, 1827, 6100, 2217, 2218, 5054, 2127, 2038, 2303, 2856, 2264, 2051, 2430, 2430,
                         3422, 3922, 2625, 1096, 6100, 1519, 3535, 2158, 2191, 2156, 2487]

        super(MSLDataset, self).__init__('MSL', msl_channels, msl_train_lens, msl_test_lens,
                                         data_path=data_path, channel_id=channel_id, training=training,
                                         download=download)

    @property
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return 55
