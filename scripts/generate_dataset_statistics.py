"""
This script generates the basic statistics and plots for a dataset. It saves the results in
resources/datasets/<dataset>/<train/test>/. The basic statistics are collected in statistics.json.
"""

import argparse

import json
import os

from timesead.data.dataset import BaseTSDataset
from timesead.utils.metadata import PROJECT_ROOT
from timesead.utils.sys_utils import check_path
from timesead.utils.utils import str2cls
from timesead.utils.plot_utils import set_style
from timesead.plots.dataset_plots import plot_mean_distribution, plot_anomaly_position_distribution, plot_anomaly_length_distribution
from timesead.data.statistics import compute_feature_statistics, compute_anomaly_positions, compute_anomaly_lengths, \
    compute_total_time_steps


def generate_statistics(dataset : BaseTSDataset, logpath : str, resolution: int = 100):

    # --- General statistics ---
    stats = {
        'n_features' : dataset.num_features,
        'n_samples' : len(dataset),
        'total_time_steps' : compute_total_time_steps(dataset)
    }

    # --- Means and standard deviations
    plot_mean_distribution(dataset, logpath, global_only=True)

    # --- Constant features ---
    normal_means, normal_stds, anomaly_means, anomaly_stds, _, _ = compute_feature_statistics(dataset)

    stats['n_constant_features_normal']  = 0 if normal_stds is None else len([i for i in normal_stds if i == 0])
    stats['n_constant_features_anomaly'] = 0 if anomaly_stds is None else len([i for i in anomaly_stds if i == 0])

    # --- Anomalies ---
    if anomaly_means is not None:

        # --- Distribution of relative anomaly positions in each time series ---
        plot_anomaly_position_distribution(dataset, logpath, resolution, yticks=5)

        # --- Distribution of lengths of anomaly intervals in the dataset ---
        plot_anomaly_length_distribution(dataset, logpath, resolution, yticks=5)

        # --- Anomaly statistics
        positions = compute_anomaly_positions(dataset)
        lengths   = compute_anomaly_lengths(dataset)

        stats['total_anomalous_points'] = len(positions)
        stats['total_anomalies']        = len(lengths)

    # --- Write statistics to file ---
    with open(os.path.join(logpath, 'statistics.json'), 'w') as fp:
        json.dump(stats, fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='timesead.data.swat_dataset.SWaTDataset')
    args = parser.parse_args()

    dataset = str2cls(args.dataset)
    logpath = os.path.join(PROJECT_ROOT, 'resources', 'datasets', args.dataset.split('.')[-1])

    set_style()

    if args.dataset == 'timesead.data.smd_dataset.SMDDataset':

        for server_id in range(28):

            path = f'{logpath}{server_id:02}'

            check_path(path)
            check_path(os.path.join(path, 'train'))
            check_path(os.path.join(path, 'test'))

            generate_statistics(dataset(server_id=server_id, training=True), os.path.join(path, 'train'))
            generate_statistics(dataset(server_id=server_id, training=False), os.path.join(path, 'test'))

    elif args.dataset == 'timesead.data.exathlon_dataset.ExathlonDataset':

        for app_id in [1, 2, 3, 4, 5, 6, 9, 10]:

            path = f'{logpath}{app_id:02}'

            check_path(path)
            check_path(os.path.join(path, 'train'))
            check_path(os.path.join(path, 'test'))

            generate_statistics(dataset(app_id=app_id, training=True), os.path.join(path, 'train'))
            generate_statistics(dataset(app_id=app_id, training=False), os.path.join(path, 'test'))


    else:

        check_path(logpath)
        check_path(os.path.join(logpath, 'train'))
        check_path(os.path.join(logpath, 'test'))

        generate_statistics(dataset(training=True), os.path.join(logpath, 'train'))
        generate_statistics(dataset(training=False), os.path.join(logpath, 'test'))
