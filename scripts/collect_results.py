import json
import os.path
import sys

import pandas as pd
from sacred.serializer import restore

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'experiments')))
from experiment_utils import remove_sacred_garbage
from utils.metadata import LOG_DIRECTORY


def collect_scores(frame, exp_path, exp_id):
    info_file = os.path.join(exp_path, exp_id, 'info.json')
    with open(info_file, mode='r') as f:
        info = json.load(f)
    # Because of sacred and jsonpickle we need to remove garbage twice!
    info = remove_sacred_garbage(info)
    info = restore(info)
    info = remove_sacred_garbage(info)

    config_file = os.path.join(exp_path, exp_id, 'config.json')
    with open(config_file, mode='r') as f:
        config = json.load(f)
    config = restore(config)

    exp_name = config['params']['training_experiment']
    exp_name_short = exp_name.rsplit('.', maxsplit=1)[1]

    print(f'Processing {exp_name_short}, id {exp_id}...')

    row = {f'Test {k}': v for k, v in info['final_scores'].items()}
    row.update(ID=exp_id, name=exp_name_short.replace('train_', ''))
    row = pd.DataFrame([row])
    row = row.set_index('ID')

    test_folds = config['params']['test_folds']

    val_scores = [{f'Validation {k}': v['score'] for k, v in info[fold]['best_val_scores'].items()}
                  for fold in [f'fold_{i}' for i in range(test_folds)]]
    df = pd.DataFrame(val_scores)
    df = df.describe().loc['mean':'mean']
    df['ID'] = [exp_id]
    df = df.set_index('ID')
    row = row.merge(df, left_index=True, right_index=True)

    frame.append(row)
    print('Done!')


def main(args):
    # First read all grid_search Experiments from the log dir
    frame = []
    for folder in args.folders:
        grid_search_root = os.path.join(args.log_dir, folder)
        if args.experiments is None:
            experiments = list(os.listdir(grid_search_root))
        else:
            experiments = args.experiments

        for entry in experiments:
            if entry.startswith('_'):
                continue

            # Collect Information from every experiment
            collect_scores(frame, grid_search_root, entry)

    frame = pd.concat(frame)

    # Move name column to front
    cols = list(frame.columns)
    cols.remove('name')
    cols.insert(0, 'name')
    frame = frame.loc[:, cols]

    # Save
    frame.to_csv(args.out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default=LOG_DIRECTORY)
    parser.add_argument('--out-file', type=str, default=os.path.join(LOG_DIRECTORY, 'metrics.csv'))
    parser.add_argument('--experiments', type=str, default=None, nargs='+')
    parser.add_argument('--folders', type=str, nargs='+', default=['grid_search', 'grid_search2'])

    main(parser.parse_args())
