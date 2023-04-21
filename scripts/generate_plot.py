import argparse

import matplotlib.pyplot as plt
import json
import os

import torch

import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))
from utils.metadata import PROJECT_ROOT
from utils.sys_utils import check_path
from utils.utils import str2cls
from utils.plot_utils import plot_error_bars, save_figure
from data.statistics import compute_feature_statistics, compute_anomaly_positions, compute_anomaly_lengths


def feature_distribution(dataset, ax=None, sort=False):

    normal_means, normal_stds, anomaly_means, anomaly_stds, _, _ = compute_feature_statistics(dataset)

    if sort:
        if anomaly_means is None:
            normal_means, normal_stds = zip(*sorted(zip(normal_means, normal_stds), key=lambda x: x[1], reverse=True))
        else:
            normal_means, normal_stds, anomaly_means, anomaly_stds = zip(*sorted(zip(normal_means, normal_stds, anomaly_means, anomaly_stds), key=lambda x: x[1], reverse=True))

    plot_error_bars(normal_means, normal_stds, ax, zorder=100)

    if anomaly_means is not None:
        plot_error_bars(anomaly_means, anomaly_stds, ax, zorder=0)

    return normal_means, normal_stds, anomaly_means, anomaly_stds


def anomaly_positions(dataset, resolution=100):

    normal_means, normal_stds, anomaly_means, anomaly_stds, _, _ = compute_feature_statistics(dataset)

    if anomaly_means is not None:

        positions = compute_anomaly_positions(dataset)

        ax = plt.gca()

        ax.hist(positions, resolution, (0, 1), histtype='stepfilled')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().set_ticks([0, 1])


def anomaly_length_disitribution(dataset, resolution=100):

    normal_means, normal_stds, anomaly_means, anomaly_stds, _, _ = compute_feature_statistics(dataset)

    if anomaly_means is not None:

        lengths = compute_anomaly_lengths(dataset)

        ax = plt.gca()

        ax.hist(lengths, resolution, (0, max(lengths)), histtype='stepfilled')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().set_ticks([0, max(lengths)])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data.swat_dataset.SWaTDataset')
    parser.add_argument('--plot', nargs='+', default=['scripts.generate_plot.feature_distribution', 'scripts.generate_plot.anomaly_positions', 'scripts.generate_plot.anomaly_length_disitribution'])
    parser.add_argument('--preprocess', nargs='+', default=[])
    parser.add_argument('-ss', '--stylesheet', nargs='+', default=[os.path.join(PROJECT_ROOT, 'resources', 'style', 'fd_style.mplstyle') for _ in range(3)])
    parser.add_argument('--save', nargs='+', default=['swat_fd', 'swat_ap', 'swat_ld'])
    args = parser.parse_args()

    assert len(args.plot) == len(args.stylesheet)
    assert len(args.plot) == len(args.save)

    dataset = str2cls(args.dataset)

    logpath = os.path.join(PROJECT_ROOT, 'resources', 'plots')

    check_path(logpath)

    plot_fns = [str2cls(plot_fn) for plot_fn in args.plot]
    prep_fns = [str2cls(prep_fn) for prep_fn in args.preprocess]

    for i in range(len(plot_fns) - len(prep_fns)):
        prep_fns.append(None)

    for style, prep_fn, plot_fn, save in zip(args.stylesheet, prep_fns, plot_fns, args.save):

        with plt.style.context(style):

            train_data = prep_fn(dataset(training=True)) if prep_fn is not None else dataset(training=True)
            test_data  = prep_fn(dataset(training=False)) if prep_fn is not None else dataset(training=False)

            plot_fn(train_data)
            save_figure(os.path.join(logpath, f'{save}_train.png'))

            plot_fn(test_data)
            save_figure(os.path.join(logpath, f'{save}_test.png'))


