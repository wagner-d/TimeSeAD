import os
import sys
import json
from typing import Tuple

import pandas as pd

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))
from utils.metadata import LOG_DIRECTORY


def get_training_time(training_experiment: str, id: str) -> Tuple[float, float]:
    experiment_root = os.path.join(LOG_DIRECTORY, training_experiment, id)

    metrics_file = os.path.join(experiment_root, 'metrics.json')
    with open(metrics_file, mode='r') as f:
        metrics = json.load(f)

    train_times = metrics['train_time']['values']
    avg_train_time = sum(train_times) / len(train_times)
    val_times = metrics['val_time']['values']
    avg_val_time = sum(val_times) / len(val_times)

    return avg_train_time, avg_val_time


def main(args):
    table = []
    names = []

    # First read all grid_search Experiments from the log dir
    for folder in args.folders:
        grid_search_root = os.path.join(LOG_DIRECTORY, folder)

        if args.experiments is None:
            experiments = [entry for entry in os.listdir(grid_search_root) if not entry.startswith('_')]
        else:
            experiments = args.experiments
        for entry in experiments:
            # Get the training times from the corresponding training experiments
            # Check how many experiments were run in parallel
            config_file = os.path.join(grid_search_root, entry, 'config.json')
            with open(config_file, mode='r') as f:
                config = json.load(f)

            experiment_name = config['params']['training_experiment']
            experiment_name = experiment_name.rsplit('.', maxsplit=1)[1]

            print(f'Extracting times for ID {entry}, {experiment_name}')

            info_file = os.path.join(grid_search_root, entry, 'info.json')
            with open(info_file, mode='r') as f:
                info = json.load(f)

            avg_train_time = 0
            avg_val_time = 0
            avg_eval_time = 0
            avg_detection_time = 0
            avg_metric_time = 0
            folds = [value for key, value in info.items() if key.startswith('fold_')]
            if len(folds) == 0:
                folds = [info]
            for fold in folds:
                training_experiments = fold['training_experiments']
                for training_experiment in training_experiments:
                    training_time, validation_time = get_training_time(experiment_name, training_experiment['train_id'])
                    avg_train_time += training_time / len(training_experiments) / len(folds)
                    avg_val_time += validation_time / len(training_experiments) / len(folds)
                    if 'evaluation_time' in training_experiment:
                        avg_eval_time += training_experiment['evaluation_time'] / len(training_experiments) / len(folds)
                    else:
                        avg_detection_time += training_experiment['detection_time'] / len(training_experiments) / len(folds)
                        avg_metric_time += training_experiment['metric_time'] / len(training_experiments) / len(folds)

            parallel_experiments = config['params']['exp_processes']

            table.append([avg_train_time, avg_val_time, avg_eval_time, avg_detection_time, avg_metric_time,
                          parallel_experiments])
            names.append(experiment_name)

    data = pd.DataFrame(table, index=names,
                        columns=['avg_train_time', 'avg_val_time', 'avg_eval_time', 'avg_detection_time',
                                 'avg_metric_time', 'parallel_experiments'])
    if (data['avg_eval_time'] == 0).all():
        data['avg_eval_time'] = data['avg_detection_time'] + data['avg_metric_time']
    data['time_per_run'] = (data['avg_train_time'] + data['avg_val_time']) * args.epochs + args.folds * data['avg_eval_time']
    data['max_grid_search_size'] = args.budget / data['time_per_run'] * data['parallel_experiments']

    data.to_csv(os.path.join(LOG_DIRECTORY, 'recon_results.csv'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=24 * 60 * 60)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--experiments', type=str, default=None, nargs='+')
    parser.add_argument('--folders', type=str, nargs='+', default=['grid_search', 'grid_search2'])

    main(parser.parse_args())
