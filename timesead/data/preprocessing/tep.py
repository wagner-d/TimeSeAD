import logging
import os
from typing import List

import pandas as pd
import pyreadr

from .common import save_statistics
from timesead.utils.utils import ceil_div

_logger = logging.getLogger(__name__)


def make_chunks(liste, chunksize):
    n_chunks = ceil_div(len(liste), chunksize)
    for i in range(n_chunks):
        start = i*chunksize
        end = min(start + chunksize, len(liste))
        yield liste[start:end]


def _split_tep_stats(processed_dir: str, chunksize: int = 100000,
                     faults: List[int] = list(range(1, 21)), fault_chunksize: int = 7,
                     training: bool = False, gzip: bool = False):
        """
        Splits the preprocessed tep stats file into one file for each fault. This is so that they an be loaded
        selectively, possibly saving RAM.
        """
        fault_str = 'Faulty'
        test_str = 'Training' if training else 'Testing'
        fname = f'TEP_{fault_str}_{test_str}.csv.gz'

        for f_chunk in make_chunks(faults, fault_chunksize):
            _logger.info(f'Reading Faults {f_chunk}...' )
            data_chunks = pd.read_csv(os.path.join(processed_dir, fname), chunksize=chunksize)
            df = []
            for data in data_chunks:
                # Filter only the needed fault
                data = data.loc[data['faultNumber'].isin(f_chunk)]

                df.append(data.copy())

            df = pd.concat(df)
            _logger.info('Done reading chunks!')

            for f in f_chunk:
                _logger.info(f'Writing data for fault {f:02d}...')
                df_f = df.loc[df['faultNumber'] == f]
                file = f'TEP_{fault_str}_{test_str}_{f:02d}.csv'
                if gzip:
                    file += '.gz'
                df_f.to_csv(os.path.join(processed_dir, file), index=False)
                _logger.info('Done!')


def _generate_tep_stats(dataset_dir: str, out_dir: str, filename: str, chunksize: int = 100000, gzip: bool = True):
    data_file = os.path.join(dataset_dir, filename)
    data = pyreadr.read_r(data_file)
    data = data[next(iter(data.keys()))]
    file_info = list(os.path.splitext(os.path.basename(data_file)))
    file_info[1] = '.csv'
    if gzip:
        file_info[1] += '.gz'
    os.makedirs(out_dir, exist_ok=True)

    # Save dataset statistics
    stats_file = os.path.join(out_dir, f'{file_info[0]}_stats.npz')
    save_statistics(data[data.columns[3:]], stats_file)

    out_file = os.path.join(out_dir, ''.join(file_info))
    assert out_file != data_file
    _logger.info(f'Saving converted file in "{out_file}"!')
    data.to_csv(out_file, index=False, chunksize=chunksize)


def preprocess_tep_data(dataset_dir: str, out_dir: str, chunksize: int = 100000, fault_chunksize: int = 7):
    """
    Preprocess TEP dataset for experiments

    :param dataset_dir: Path to the dataset folder
    :param out_dir: Directory where the preprocessed data should be saved. This directory should exist already.
    :param chunksize: The processed csv is saved using this chunksize value.
    :param fault_chunksize: The processed TEP faults as saved in chunks specified by this.
    """
    for file in ['TEP_FaultFree_Testing.RData', 'TEP_Faulty_Testing.RData', 'TEP_FaultFree_Training.RData',
                 'TEP_Faulty_Training.RData']:
        gzip = 'Faulty' in file
        _generate_tep_stats(dataset_dir, out_dir, file, chunksize=chunksize, gzip=gzip)

    _split_tep_stats(out_dir, chunksize, fault_chunksize=fault_chunksize, training=True, gzip=False)
    _split_tep_stats(out_dir, chunksize, fault_chunksize=fault_chunksize, training=False, gzip=False)
