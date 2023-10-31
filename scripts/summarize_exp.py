'''
Summarize experiment scores for different seeds and datasets to a single json file. 
Also checks for varying best parameters in different seeds of the same run.

This expects the logs to be structured as results/<dataset>/<experiment>_<seed>
'''
# TODO: Generalize for any log structure 

import os 
import sys
import glob
import json
import numpy as np
import logging
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

def _get_changing_params_impl(changing_params: dict, context: list, param1: object, param2: object):
    # Recursively check if params are different and update changing_params 
    if isinstance(param1, dict):
        for key in param1:
            new_context = context[:]
            new_context.append(key)
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


def collect_dataset_results(summary: dict, dataset_dir: str, experiment: str):
    # Read result files for the dataset, experiment and store scores in summary dict
    scores = dict()
    params = []
    # For each <exp>_*/info.json in the dataset dirs
    info_files = glob.glob(os.path.join(args.results_dir, dataset_dir, f'{experiment}_*', 'info.json'))
    if len(info_files) < 3:
        logging.warning(f'There should be atleast 3 runs with different seeds. Got only {len(info_files)}')
    for info_file in info_files:
        with open(info_file) as ff:
            logging.debug(f'Parsing {info_file}')
            data = json.load(ff)
            # Collect results from "final_scores" entry
            for metric, val in data['final_scores'].items():
                if metric not in scores:
                    scores[metric] = []
                scores[metric].append(val)
            # Collect unique best parameters from "final_best_params"
            if data['final_best_params'] not in params:
                params.append(data['final_best_params'])

    changing_params = dict()
    if len(params) > 1:
        changing_params = get_changing_params(params)
        logging.warning(f'More than one best params for {dataset_dir}, {changing_params}')

    summary[dataset_dir] = dict()
    summary_dataset = summary[dataset_dir]
    # Calculate mean, std of the results
    summary_dataset['scores'] = dict()
    for metric, vals in scores.items():
        summary_dataset['scores'][metric] = (np.mean(vals), np.std(vals))
    summary_dataset['changing_params'] = changing_params
    logging.debug(f'Done with {dataset_dir}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='Experiment name to summarize results of', type=str)
    parser.add_argument('--results_dir', help='The directory holding the results', type=str, default='results')
    parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset_dirs = []
    for entry in os.scandir(args.results_dir):
        if not entry.is_dir():
            continue
        dataset_dirs.append(entry.name)
    logging.debug(f'Found datasets: {dataset_dirs}')

    # summary  dataset_dir:{scores, params}
    summary = dict()
    for dataset_dir in dataset_dirs:
        try: 
            collect_dataset_results(summary, dataset_dir, args.experiment)
        except Exception as ee:
            logging.error(f'Error processing results for {args.experiment} in {dataset_dir}: {ee}')
        
    # Write the dict as json to <exp>_summary.json
    results_file = os.path.join(args.results_dir, f'{args.experiment}_summary.json')
    with open(results_file, 'w') as summary_file:
        json.dump(summary, summary_file, indent=2)
    logging.info(f'Wrote summary to {results_file}')

    # Log summary of collected scores
    for dataset_dir, data in summary.items():
        logging.debug(f'\n=== {dataset_dir} ===')
        for metric, vals in data['scores'].items():
            logging.debug(f'{metric}: {vals[0]:.2f}\u00B1{vals[1]:.2f}')

