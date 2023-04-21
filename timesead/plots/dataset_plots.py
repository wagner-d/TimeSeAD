from typing import Optional, List, Union

import torch
import os

import matplotlib.pyplot as plt

from timesead.data.statistics import compute_anomaly_positions, compute_anomaly_lengths, compute_feature_statistics
from timesead.utils.plot_utils import plot_histogram, save_plot, plot_sequence_against_anomaly, plot_error_bars
from timesead.utils.sys_utils import check_path


def plot_features_against_anomaly(dataset : torch.utils.data.Dataset, path : str, xticks : Optional[int] = None,
                                  shape : str = 'tf', interval_size : Optional[int] = None, scatter : bool = True):

    for i in range(len(dataset)):

        data, targets = dataset[i]
        data, targets = data[0], targets[0]

        data = data.permute((shape.index('f'), shape.index('t')))

        for feature in range(data.shape[0]):

            plot_sequence_against_anomaly(values=data[feature].tolist(), targets=targets.tolist(), xticks=xticks,
                                          scatter=scatter)
            save_plot(os.path.join(path, f'ts{i:03}_f{feature:03}.png'))

            if interval_size:

                check_path(os.path.join(path, 'intervals'))

                for interval_index in range((data.shape[1] // interval_size) + int(data.shape[1] % interval_size > 0)):

                    plot_sequence_against_anomaly(values=data[feature][interval_index * interval_size: (interval_index + 1) * interval_size].tolist(),
                                                  targets=targets[interval_index * interval_size: (interval_index + 1) * interval_size].tolist(),
                                                  xticks=xticks, scatter=scatter, yticks=False)
                    save_plot(os.path.join(path, 'intervals', f'ts{i:03}_f{feature:03}_int{interval_index:03}.png'))


def plot_anomaly_distribution(dataset : torch.utils.data.Dataset, path : str, resolution : int = 100,
                              yticks : Optional[Union[int, List]] = None, **kwargs):

    positions = compute_anomaly_positions(dataset)

    plot_histogram(positions, resolution, yticks, **kwargs)
    save_plot(os.path.join(path, 'anomaly_position_distribution.png'))


def plot_anomaly_length_distribution(dataset : torch.utils.data.Dataset, path : str, resolution : int = 100,
                                     yticks : Optional[Union[int, List]] = None, **kwargs):

    lengths = compute_anomaly_lengths(dataset)

    plot_histogram(lengths, resolution, yticks, hist_range=(0, max(lengths)), xticks=[0, max(lengths)], **kwargs)
    save_plot(os.path.join(path, 'anomaly_lengths_distribution.png'))


def plot_anomaly_position_distribution(dataset : torch.utils.data.Dataset, path : str, resolution : int = 100,
                                       yticks : Optional[Union[int, List]] = None, **kwargs):

    positions = compute_anomaly_positions(dataset)

    plot_histogram(positions, resolution, yticks, xticks=[0, 1], hist_range=(0, 1))
    save_plot(os.path.join(path, 'anomaly_position_distribution.png'))


def plot_mean_distribution(dataset : torch.utils.data.Dataset, path : str, interval_size : Optional[int] = None,
                           shape : str = 'tf', yticks : Optional[Union[List[int], List[float]]] = None,
                           global_only : bool = False):

    normal_means, normal_stds, anomaly_means, anomaly_stds, _, _ = compute_feature_statistics(dataset)

    if not anomaly_means is None:

        plot_error_bars(anomaly_means, anomaly_stds, offset=0.1, fmt='none', zorder=20, ecolor='r')
        plt.scatter([i + 0.1 for i, _ in enumerate(anomaly_stds) if _ == 0], [0 for _ in anomaly_stds if _ == 0],
                    marker='d', color='r', s=5, zorder=0)

    if not normal_means is None:

        plot_error_bars(normal_means, normal_stds, offset=-0.1, fmt='none', zorder=30, ecolor='k')
        plt.scatter([i - 0.1 for i, _ in enumerate(normal_stds) if _ == 0], [0 for _ in normal_stds if _ == 0],
                    marker='d', color='k', s=5, zorder=10)

    if yticks:
        plt.gca().get_yaxis().set_ticks(yticks)

    save_plot(os.path.join(path, 'feature_distribution.png'))

    if not global_only:

        for i in range(len(dataset)):

            data, targets = dataset[i]
            data, targets = data[0], targets[0]

            data = data.permute((shape.index('t'), shape.index('f')))

            if interval_size:
                for interval_index in range((data.shape[0] // interval_size) + int(data.shape[0] % interval_size > 0)):

                    data_interval   = data[interval_index * interval_size : (interval_index + 1) * interval_size]
                    target_interval = targets[interval_index * interval_size : (interval_index + 1) * interval_size]

                    normal_means, normal_stds, anomaly_means, anomaly_stds = None, None, None, None

                    if torch.sum(target_interval) < target_interval.shape[0]:

                        normal_means = torch.mean(data_interval[target_interval == 0], 0)
                        normal_stds  = torch.std(data_interval[target_interval == 0], 0)

                    if torch.sum(targets) > 0:

                        anomaly_means = torch.mean(data_interval[target_interval == 1], 0)
                        anomaly_stds  = torch.std(data_interval[target_interval == 1], 0)

                    if not anomaly_means is None:

                        plot_error_bars(anomaly_means, anomaly_stds, offset=-0.1, fmt='none', zorder=20)
                        plt.scatter([i + 0.1 for i, _ in enumerate(anomaly_stds) if _ == 0],
                                    [0 for _ in anomaly_stds if _ == 0],
                                    marker='d', color='r', s=5, zorder=0)

                    if not normal_means is None:

                        plot_error_bars(normal_means, normal_stds, offset=0.1, fmt='none', zorder=30)
                        plt.scatter([i - 0.1 for i, _ in enumerate(normal_stds) if _ == 0],
                                    [0 for _ in normal_stds if _ == 0],
                                    marker='d', color='k', s=5, zorder=10)

                    save_plot(os.path.join(path, f'ts{i:03}_int{interval_index:03}.png'))

            else:

                normal_means, normal_stds, anomaly_means, anomaly_stds = None, None, None, None

                if torch.sum(targets) < targets.shape[0]:

                    normal_means = torch.mean(data[targets == 0], 0)
                    normal_stds  = torch.std(data[targets == 0], 0)

                if torch.sum(targets) > 0:

                    anomaly_means = torch.mean(data[targets == 1], 0)
                    anomaly_stds  = torch.std(data[targets == 1], 0)

                if not anomaly_means is None:

                    plot_error_bars(anomaly_means, anomaly_stds, offset=0.1, fmt='none', zorder=20, ecolor='r')
                    plt.scatter([i + 0.1 for i, _ in enumerate(anomaly_stds) if _ == 0],
                                [0 for _ in anomaly_stds if _ == 0],
                                marker='d', color='r', s=5, zorder=0)

                if not normal_means is None:

                    plot_error_bars(normal_means, normal_stds, offset=-0.1, fmt='none', zorder=30, ecolor='k')
                    plt.scatter([i - 0.1 for i, _ in enumerate(normal_stds) if _ == 0],
                                [0 for _ in normal_stds if _ == 0],
                                marker='d', color='k', s=5, zorder=10)

                if yticks:
                    plt.gca().get_yaxis().set_ticks(yticks)

                save_plot(os.path.join(path, f'ts{i:03}.png'))


