'''
Summarize experiment scores for different seeds and datasets to a single json file. 
Also checks for varying best parameters in different seeds of the same run.
'''

import os 
import sys
import glob
import json
import numpy as np
import logging
import argparse

from timesead.utils.metadata import LOG_DIRECTORY

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

def _get_changing_params_impl(changing_params: dict, context: list, param1: object, param2: object):
    # Recursively check if params are different and update changing_params 
    if isinstance(param1, list):  # Make it hashable if not
        param1 = tuple(param1)
        param2 = tuple(param2)

    if isinstance(param1, dict):
        for key in param1:
            new_context = context[:]
            new_context.append(key)
            if not isinstance(param2, dict) or key not in param2:
                _get_changing_params_impl(changing_params, new_context, param1[key], None)
            else:
                _get_changing_params_impl(changing_params, new_context, param1[key], param2[key])
    elif param1 != param2:
        param_str = '.'.join(context)
        if param_str not in changing_params:
            changing_params[param_str] = set()
        changing_params[param_str].add(param1)
        changing_params[param_str].add(param2)

def get_changing_params(params: list):
    # Take list of params dict and return dict {<changing_param>: list of values}
    changing_params = dict()
    init_param = params[0]
    for param in params[1:]:
        for key in param:
            _get_changing_params_impl(changing_params, [key], init_param[key], param[key])
    # Change dict of set values to list
    for key, value in changing_params.items():
        changing_params[key] = list(value)
    return changing_params


def collect_results(summary: dict, log_dir: str):
    # Aggregate information from logs to summary dict per (dataset, experiment)
    logging.info(f'Parsing dir {log_dir}')
    exp_summary = None

    with open(os.path.join(log_dir, 'config.json')) as ff:
        data = json.load(ff)
        dataset = data['dataset']['name'][:-7]  # Remove 'Dataset'
        if dataset == 'SMD':  # SMD has additional server_id information
            dataset = f'{dataset}_{data["dataset"]["ds_args"]["server_id"]}'
        # training_experiment string is of the form module.train_<experiment>
        experiment = data['params']['training_experiment'].split('train_')[-1] 

        if (dataset, experiment) not in summary:
            summary[(dataset, experiment)] = {'seeds': [], 'scores': {}, 'params': []}
        exp_summary = summary[(dataset, experiment)]

        seed = data['seed']
        if seed in exp_summary['seeds']:
            logging.warning(f'Multiple runs with the same seed. Skipping log {log_dir}')
            return
        exp_summary['seeds'].append(seed)

    # Scores and parameter information in info.json
    with open(os.path.join(log_dir, 'info.json')) as ff:
        data = json.load(ff)
        scores = exp_summary['scores']
        for metric, val in data['final_scores'].items():
            if metric not in scores:
                scores[metric] = []
            scores[metric].append(val)
        params = exp_summary['params']
        if data['final_best_params'] not in params:
            params.append(data['final_best_params'])


def summarize_results(summary: dict):
    # Consolidate aggregated scores and params for each dataset experiment
    for dataset_exp, data in summary.items():
        logging.debug(f'Summarizing info for {dataset_exp}')
        scores = dict()
        for metric, vals in data['scores'].items():
            scores[metric] = (np.mean(vals), np.std(vals))
        data['scores'] = scores
        
        data['changing_params'] = get_changing_params(data['params'])
        data['params'] = data['params'][0]


def summary_to_json_dict(summary: dict):
    # Convert the summary dict to a json dump format
    json_data = []
    for (dataset, experiment), data in summary.items():
        entry = {'dataset': dataset, 'experiment': experiment}
        entry.update(data)
        json_data.append(entry)
    return json_data
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', help='Glob pattern for log folders', type=str, default=os.path.join(LOG_DIRECTORY, 'grid_search', '*'))
    parser.add_argument('--output_file', help='JSON output file name', type=str, default='results/summary.json')
    parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log_dirs = glob.glob(args.log_dir)
    log_dirs.sort(key=os.path.getmtime, reverse=True)  # Start with latest logs
    logging.debug(f'Found logs: {log_dirs}')

    # Summary  (dataset, experiment): {scores: [], seeds: [], params: []}
    summary = dict()
    for log_dir in log_dirs:
        try:
            collect_results(summary, log_dir)
        except Exception as ee:
            logging.error(f'Error processing results in log {log_dir}: {ee}')

    summarize_results(summary)
    json_data = summary_to_json_dict(summary)

    with open(args.output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
    logging.info(f'Wrote summary to {args.output_file}')

