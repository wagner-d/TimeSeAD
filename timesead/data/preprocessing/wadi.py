import logging
import os

import numpy as np
import pandas as pd

from .common import save_statistics

_logger = logging.getLogger(__name__)


def preprocess_wadi_data(dataset_dir: str, out_dir: str):
    """
    Preprocess WaDI dataset for experiments

    :param dataset_dir: Path to the dataset folder
    :param out_dir: Directory where the preprocessed data should be saved. This directory should exist already.
    """
    file = os.path.join(dataset_dir, 'WADI_14days_new.csv')

    _logger.info(f'Reading file "{file}"!')
    data = pd.read_csv(file)
    data = data.drop(columns=['Row', 'Date', 'Time', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS',
                                '2_P_002_STATUS'])
    nan_rows, = np.nonzero(data.isna().any(axis=1).to_numpy())
    for i in nan_rows:
        row = data.iloc[i]
        nonzero, = np.nonzero(row.isna().to_numpy())
        data.iloc[i, nonzero] = data.iloc[i - 1, nonzero]

    file_info = list(os.path.splitext(os.path.basename(file)))

    # Save dataset statistics
    stats_file = os.path.join(out_dir, f'{file_info[0]}_stats.npz')
    save_statistics(data, stats_file)
