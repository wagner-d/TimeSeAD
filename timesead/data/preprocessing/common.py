from typing import Tuple, Dict, Union
import pandas as pd
import numpy as np


def update_statistics_increment(frame: pd.DataFrame, mean: np.ndarray = None, min_val: np.ndarray = None,
                                max_val: np.ndarray = None, old_n: int = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Incrementally compute feature-wise mean, minimum, and maximum values for a dataset consisting of multiple
    :class:`~pandas.DataFrame`\s.

    :param frame: The current data with wich the statistics should be updated.
    :param mean: Mean value returned by previous calls to the function or `None` if this is the first call.
    :param min_val: Minimum value returned by previous calls to the function or `None` if this is the first call.
    :param max_val: Maximum value returned by previous calls to the function or `None` if this is the first call.
    :param old_n: Total number of data points processed in earlier calls to the function or `None` if this is the first
        call.
    :return: A tuple `(mean, min_val, max_val, old_n + n)` that contains the updated statistics and the total number of
        processed data points.
    """
    n = frame.shape[0]
    if mean is not None:
        mean = (old_n / (old_n + n)) * mean + n / (old_n + n) * frame.mean().to_numpy()
    else:
        mean = frame.mean().to_numpy()

    if min_val is not None:
        min_val = np.minimum(min_val, frame.min().to_numpy())
    else:
        min_val = frame.min().to_numpy()

    if max_val is not None:
        max_val = np.maximum(max_val, frame.max().to_numpy())
    else:
        max_val = frame.max().to_numpy()

    return mean, min_val, max_val, old_n + n


def save_statistics(frame: pd.DataFrame, path: str):
    """
    Compute feature-wise mean, standard deviation, minimum, and maximum values for a dataset consisting of a single
    :class:`~pandas.DataFrame` and save them as a `.npz` file.

    :param frame: The dataset for which to compute and save statistics.
    :param path: Path to save the statistics via :func:`numpy.savez`.
    """
    mean = frame.mean().to_numpy()
    std = frame.std().to_numpy()
    min = frame.min().to_numpy()
    max = frame.max().to_numpy()
    median = frame.median().to_numpy()

    np.savez(path, mean=mean, std=std, min=min, max=max, median=median)


def minmax_scaler(frame: Union[pd.DataFrame, np.ndarray], stats: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Scale features in a dataset independently to the [0, 1] range using pre-computed minimum and maximum values.

    :param frame: The dataset to scale. This should either be a :class:`~pandas.DataFrame` or a :class:`~numpy.ndarray`
        of shape `(N*, D)`.
    :param stats: A dictionary that contains pre-computed feature-wise minimum and maximum values as
        :class:`~numpy.ndarray`\s of shape `(D,)` in the keys 'min' and 'max', respectively.
    :return: The scaled :class:`~pandas.DataFrame` or a :class:`~numpy.ndarray`. Note that this will usually be a copy
        as this function does not modify any values in place.
    """
    min = stats['min']
    max = stats['max']
    range = max - min
    # Fix (near-)constant features
    constant_mask = range < 10 * np.finfo(np.float32).eps
    range[constant_mask] = 1

    return (frame - min) / range


def standard_scaler(frame: Union[pd.DataFrame, np.ndarray], stats: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Scale features in a dataset such that they have zero mean and unit standard deviation using pre-computed mean and
    standard deviation values.

    :param frame: The dataset to scale. This should either be a :class:`~pandas.DataFrame` or a :class:`~numpy.ndarray`
        of shape `(N*, D)`.
    :param stats: A dictionary that contains pre-computed feature-wise mean and standard deviation values as
        :class:`~numpy.ndarray`\s of shape `(D,)` in the keys 'mean' and 'std', respectively.
    :return: The scaled :class:`~pandas.DataFrame` or a :class:`~numpy.ndarray`. Note that this will usually be a copy
        as this function does not modify any values in place.
    """
    mean = stats['mean']
    std = stats['std']
    # Fix (near-)constant features
    constant_mask = std < 10 * np.finfo(np.float32).eps
    std[constant_mask] = 1

    return (frame - mean) / std
