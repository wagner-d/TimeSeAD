import os
from typing import List

import numpy as np
import pandas as pd

from .common import save_statistics


def preprocess_smd_data(dataset_dir: str, out_dir: str, filenames: List[str]):
    """
    Preprocess SMD dataset for experiments

    :param dataset_dir: Path to the dataset folder
    :param out_dir: Directory where the preprocessed data should be saved. This directory should exist already.
    """
    for filename in filenames:
        data = np.genfromtxt(os.path.join(dataset_dir, 'train', filename), dtype=np.float32, delimiter=',')
        data = pd.DataFrame(data)

        file_info = filename.split('.')

        # Save dataset statistics
        stats_file = os.path.join(out_dir, f'{file_info[0]}_stats.npz')
        save_statistics(data, stats_file)
