import json
import os
from argparse import Namespace

from timesead.utils.metadata import RESOURCE_DIRECTORY, DATA_DIRECTORY


REPOSITORY_PATH = 'https://github.com/wagner-d/TimeSeAD/blob/master/'

readme = [
    'This directory contains all raw data files to be loaded.',
    '',
    '| dataset | #features | #constant | #samples | #anomalies [intervals/points/percent] | anomaly lengths | anomaly positions | feature distribution |',
    '| - | - | - | - | - | - | - | - |'
]


def generate_readme():

    for dataset in sorted(os.listdir(os.path.join(RESOURCE_DIRECTORY, 'datasets'))):

        if not os.path.isfile(os.path.join(RESOURCE_DIRECTORY, 'datasets', dataset, 'train/statistics.json')):
            continue

        with open(os.path.join(RESOURCE_DIRECTORY, 'datasets', dataset, 'train/statistics.json')) as file:

            stats = json.load(file)
            stats = Namespace(**stats)

            train_anomaly_lengths       = f'![-]({REPOSITORY_PATH}resources/datasets/{dataset}/train/anomaly_lengths_distribution.png?raw=true)' if os.path.isfile(os.path.join(RESOURCE_DIRECTORY, 'datasets', dataset, 'train/anomaly_lengths_distribution.png')) else '-'
            train_anomaly_positions     = f'![-]({REPOSITORY_PATH}resources/datasets/{dataset}/train/anomaly_position_distribution.png?raw=true)' if os.path.isfile(os.path.join(RESOURCE_DIRECTORY, 'datasets', dataset, 'train/anomaly_position_distribution.png')) else '-'
            train_feature_distributions = f'![-]({REPOSITORY_PATH}resources/datasets/{dataset}/train/feature_distribution.png?raw=true)'

            n_constant_features_anomalous = f'/{stats.n_constant_features_anomaly}(a)' if hasattr(stats, 'total_anomalies') else ''
            n_anomalies                   = f'{stats.total_anomalies}/{stats.total_anomalous_points}/{round(100 * stats.total_anomalous_points / stats.total_time_steps, 2)}%' if hasattr(stats, 'total_anomalies') else '0'

            entry = f'| {dataset}(train) ' \
                    f'| {stats.n_features} ' \
                    f'| {stats.n_constant_features_normal}(n){n_constant_features_anomalous} ' \
                    f'| {stats.n_samples}/{stats.total_time_steps} ' \
                    f'| {n_anomalies} ' \
                    f'| {train_anomaly_lengths} ' \
                    f'| {train_anomaly_positions} ' \
                    f'| {train_feature_distributions} |'

            readme.append(entry)

        with open(os.path.join(RESOURCE_DIRECTORY, 'datasets', dataset, 'test/statistics.json')) as file:

            stats = json.load(file)
            stats = Namespace(**stats)

            test_anomaly_lengths       = f'![-](https://github.com/wagner-d/TimeSeAD/blob/master/resources/datasets/{dataset}/test/anomaly_lengths_distribution.png?raw=true)' if os.path.isfile(os.path.join(RESOURCE_DIRECTORY, 'datasets', dataset, 'test/anomaly_lengths_distribution.png')) else '-'
            test_anomaly_positions     = f'![-](https://github.com/wagner-d/TimeSeAD/blob/master/resources/datasets/{dataset}/test/anomaly_position_distribution.png?raw=true)' if os.path.isfile(os.path.join(RESOURCE_DIRECTORY, 'datasets', dataset, 'test/anomaly_position_distribution.png')) else '-'
            test_feature_distributions = f'![-](https://github.com/wagner-d/TimeSeAD/blob/master/resources/datasets/{dataset}/test/feature_distribution.png?raw=true)'

            n_constant_features_anomalous = f'/{stats.n_constant_features_anomaly}(a)' if hasattr(stats, 'total_anomalies') else ''
            n_anomalies                   = f'{stats.total_anomalies}/{stats.total_anomalous_points}/{round(100 * stats.total_anomalous_points / stats.total_time_steps, 2)}%' if hasattr(stats, 'total_anomalies') else '0'

            entry = f'| {dataset}(test) ' \
                    f'| {stats.n_features} ' \
                    f'| {stats.n_constant_features_normal}(n){n_constant_features_anomalous} ' \
                    f'| {stats.n_samples}/{stats.total_time_steps} ' \
                    f'| {n_anomalies} ' \
                    f'| {test_anomaly_lengths} ' \
                    f'| {test_anomaly_positions} ' \
                    f'| {test_feature_distributions} |'

            readme.append(entry)

    with open(os.path.join(DATA_DIRECTORY, 'README.md'), 'w+') as file:

        for line in readme:
            file.write(line)
            file.write('\n')


if __name__ == '__main__':

    generate_readme()

