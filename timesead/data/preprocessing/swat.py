import logging
import os
from typing import Optional

import pandas as pd

from .common import save_statistics

_logger = logging.getLogger(__name__)


def preprocess_swat_data(dataset_dir: str, out_dir: str, chunksize: Optional[int] = None, gzip: bool = False):
    """
    Preprocess SWaT dataset for experiments

    :param dataset_dir: Path to the dataset folder
    :param out_dir: Directory where the preprocessed data should be saved. This directory should exist already.
    :param chunksize: The processed csv data uses this chunksize value if set.
    :param gzip: Whether the output file should be saved as a gzip file.
    """
    data_files = [
        os.path.join(dataset_dir, 'SWaT_Dataset_Attack_v0.xlsx'),
        os.path.join(dataset_dir, 'SWaT_Dataset_Normal_v1.xlsx'),
    ]
    for file in data_files:
        _logger.info(f'Reading file "{file}"!')
        data = pd.read_excel(file, skiprows=1)
        data = data.rename(columns={c: c.strip() for c in data.columns})
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        # Fix labeling bug
        data.loc[data['Normal/Attack'] == 'A ttack', 'Normal/Attack'] = 'Attack'

        file_info = list(os.path.splitext(os.path.basename(file)))
        file_info[1] = '.csv'
        if gzip:
            file_info.append('.gz')
        os.makedirs(out_dir, exist_ok=True)

        # Save dataset statistics
        stats_file = os.path.join(out_dir, f'{file_info[0]}_stats.npz')
        save_statistics(data[data.columns[1:-1]], stats_file)

        # Save csv file
        out_file = os.path.join(out_dir, ''.join(file_info))
        assert out_file != file
        _logger.info(f'Saving converted file in "{out_file}"!')
        data.to_csv(out_file, index=False, chunksize=chunksize)
