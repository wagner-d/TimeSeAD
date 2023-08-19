import collections.abc
import copy
import functools
import os.path
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as Pool
from typing import Tuple, List

import sacred.utils
import torch
import torch.utils.data
# from torch.multiprocessing import Pool

from timesead_experiments.utils import make_experiment, make_experiment_tempfile, data_ingredient, load_dataset, \
    SerializationGuard, run_command, serialization_guard, remove_sacred_garbage
from timesead.evaluation import Evaluator
from timesead.data.dataset import collate_fn
from timesead.utils.metadata import LOG_DIRECTORY
from timesead.utils.utils import Bunch, param_grid_to_list_of_dicts, str2cls
from timesead.utils.torch_utils import set_threads, clear_gpu_memory
from timesead.utils.rng_utils import set_seed


experiment = make_experiment(ingredients=[data_ingredient])


def train_once(config_updates, params, seed):
    set_seed(seed)
    set_threads(params.threads_per_process)
    clear_gpu_memory()

    # Run the training experiment to get a trained model
    train_exp = str2cls(f'{params.training_experiment}.experiment')
    print('training with parameters', config_updates)
    train_run = run_command(train_exp, config_updates=config_updates)
    train_id = train_run._id
    del train_run

    clear_gpu_memory()

    return train_id


def evaluate_once(train_id, config_updates, detector_param_updates, params, val_ds_params, val_split, seed):
    set_seed(seed)
    set_threads(params.threads_per_process)

    # Train for 0 epochs to get the dataset
    train_exp        = str2cls(f'{params.training_experiment}.experiment')
    dataset_run      = run_command(train_exp, command='get_datasets', config_updates=config_updates)
    train_val_loader = dataset_run.result[1]

    del dataset_run

    # Load the model from the training experiment
    exp_name_short = params.training_experiment.rsplit('.', maxsplit=1)[1]
    model_file     = os.path.join(LOG_DIRECTORY, exp_name_short, train_id, 'final_model.pth')
    saved_model    = torch.load(model_file, map_location=params.device)

    if 'detector' in saved_model:
        if hasattr(saved_model['detector'], 'model'):
            model = saved_model['detector'].model
        else:
            model = None
    else:

        new_config_updates                       = copy.deepcopy(config_updates)
        new_config_updates['training']['epochs'] = 0
        new_config_updates['train_detector']     = False

        train_run = run_command(train_exp, config_updates=new_config_updates, options={'--unobserved': True})

        model_state = saved_model['model']
        model       = train_run.result['model']

        model.load_state_dict(model_state)

        del train_run

    del saved_model

    # Load the validation set with anomaly labels
    if 'dataset' in config_updates and 'pipeline' in config_updates['dataset'] and 'pipeline' in val_ds_params:
        val_ds_params = copy.deepcopy(val_ds_params)
        sacred.utils.recursive_update(val_ds_params['pipeline'], config_updates['dataset']['pipeline'])

    val_evaluator = Evaluator()
    data          = load_dataset(**val_ds_params)
    val_ds        = data[val_split]
    val_loader    = torch.utils.data.DataLoader(val_ds, batch_size=params.batch_size, num_workers=0,
                                                collate_fn=collate_fn(params.batch_dim))

    detector_params = dict(model=model, val_loader=train_val_loader, save_detector=False)

    detector_params.update(config_updates)

    best_score           = -float('inf')
    best_detector        = None
    best_detector_params = None
    best_info            = None
    avg_detection_time   = 0
    avg_metric_time      = 0
    labels               = None
    best_detector_scores = None
    val_scores = []

    for updates in detector_param_updates:

        print('Testing detector with parameters', updates)
        detector_params.update(updates)

        start = time.perf_counter()

        detector_run = run_command(train_exp, command='get_anomaly_detector', config_updates=detector_params)
        detector     = detector_run.result
        detector     = detector.to(params.device)
        del detector_run

        detector.eval()

        labels, scores = detector.get_labels_and_scores(val_loader)

        end = time.perf_counter()

        avg_detection_time += (end - start) / len(detector_param_updates)

        start                 = time.perf_counter()
        val_score, other_info = val_evaluator.__getattribute__(params.validation_metric)(labels, scores)
        end                   = time.perf_counter()

        finite = torch.isfinite(scores)
        if not torch.all(finite):
            infinite = torch.logical_not(finite)
            print(f'Wrong params: {config_updates}, {updates}')
            print(f'Not finite: {torch.arange(len(scores), device=scores.device)[infinite]}, {scores[infinite]}')

            for name, param in detector.model.state_dict().items():
                print(f'{name} finite: {torch.all(torch.isfinite(param))}')

            raise AssertionError('Scores should be finite!')

        avg_metric_time += (end - start) / len(detector_param_updates)

        val_scores.append((config_updates, updates, val_score))

        if val_score > best_score:

            best_score           = val_score
            best_detector        = detector
            best_detector_params = updates
            best_info            = other_info
            best_detector_scores = scores

        del detector

    print(f'Result for {params.validation_metric}: {best_score}, other information: {best_info}')

    # Save best detector to a file so that the main process can load it if necessary
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.pth') as f:

        best_detector_file = f.name
        torch.save(dict(detector=best_detector), f)

    # Clear cache
    del best_detector
    clear_gpu_memory()

    best_val_scores = {params.validation_metric: dict(score=best_score, info=best_info)}
    for metric in params.evaluation_metrics:
        if metric == params.validation_metric:
            continue

        val_score, other_info = val_evaluator.__getattribute__(metric)(labels, best_detector_scores)
        best_val_scores[metric]    = dict(score=val_score, info=other_info)

    return best_detector_file, best_detector_params, best_val_scores, avg_detection_time, avg_metric_time, val_scores


def compute_val_splits(folds: int, val_fold: int, padding: int = 1) -> Tuple[List[int], int, List[int]]:
    if folds == 1:
        return [1], 0, []

    # Compute the partition of the labelled data
    splits = [val_fold - padding, min(val_fold, padding), 1, min(folds - val_fold - 1, padding), folds - val_fold - 1 - padding]

    val_fold_index = 2
    if splits[0] <= 0:
        val_fold_index -= 1
    if splits[1] <= 0:
        val_fold_index -= 1

    splits = list(filter(lambda x: x > 0, splits))

    # Position of validation fold in the partitions
    test_fault_indices = list(range(len(splits)))
    test_fault_indices.remove(val_fold_index)
    if padding > 0:
        if val_fold_index > 0:
            test_fault_indices.remove(val_fold_index - 1)
        if val_fold_index < len(splits) - 1:
            test_fault_indices.remove(val_fold_index + 1)

    return splits, val_fold_index, test_fault_indices


def evaluate_all(train_ids, updated_param_grid, detector_param_grid, params, dataset, val_fold_index, context, seed):
    eval_proc_func = functools.partial(evaluate_once, detector_param_updates=detector_param_grid, params=params,
                                       val_ds_params=dataset, val_split=val_fold_index, seed=seed)
    with Pool(max_workers=params.exp_processes, mp_context=context) as pool:
        result = pool.map(eval_proc_func, train_ids, updated_param_grid)

    best_params = None
    best_model = None
    best_id = None
    best_scores = None
    best_metric = -float('inf')
    detector_files = []
    training_experiments = []
    all_val_scores = []
    for (detector_file, detector_params, best_val_scores, detection_time, metric_time, val_scores), train_id, \
        config_updates in zip(result, train_ids, updated_param_grid):
        # Store information about the training runs
        config_updates.update(detector_params)
        training_experiments.append({
            'experiment': params.training_experiment,
            'train_id': train_id,
            'params': config_updates,
            'detection_time': detection_time,
            'metric_time': metric_time,
            **best_val_scores
        })
        detector_files.append(detector_file)

        score = best_val_scores[params.validation_metric]['score']
        info = best_val_scores[params.validation_metric]['info']

        if score > best_metric:
            best_metric = score
            best_params = config_updates
            best_model = dict(detector=detector_file, **info)
            best_id = train_id
            best_scores = best_val_scores

        all_val_scores.extend(val_scores)

    return best_model, best_params, best_metric, best_scores, best_id, training_experiments, detector_files, all_val_scores


@experiment.config
def config():
    # General Evaluation parameters
    params = dict(
        # The training experiment that will be executed. Please provide a relative path from the experiments and
        # use '.' as the separator
        #training_experiment='reconstruction.train_lstm_ae',
        training_experiment='train_model_template2',
        # This will be computed over the validation set and used to determine the best parameters in the grid
        validation_metric='best_f1_score',
        # These metrics will be calculated on the test set
        evaluation_metrics=['best_f1_score', 'auprc'],
        batch_size=128,
        device='cpu',
        # Number of processes to train in parallel and CPU threads used for each process
        exp_processes=1,
        threads_per_process=1,
        test_folds=5,
        train_ids=None,
        padding=1
    )

    # Use this to specify training parameters that should be searched over. They will take precedence over the
    # values specified in param_updates. The format for this is that you define a list of
    # possible values for each attribute. For example:
    # training_param_grid = dict(
    #     model_params=dict(
    #         hidden_dimensions=[[40], [50]]
    #     ),
    #     training=dict(
    #         optimizer = {
    #             'args': dict(lr=[1e-3, 0.01])
    #         }
    #     )
    # )
    training_param_grid = dict()

    # Use this to overwrite training parameters with a single value. Will be overwritten with the values in
    # training_param_grid
    training_param_updates = dict()

    # This should contain all values for the detector parameters that should be searched over.
    # Note that this should contain only parameters of the detector, for which no retraining is needed.
    detector_param_grid = dict()


@data_ingredient.config
def data_config():
    ds_args = dict(
        training=False
    )
    split = (0.3, 0.7)


@experiment.automain
@serialization_guard
def main(params, training_param_grid, training_param_updates, detector_param_grid, dataset, _run, _seed):

    params = dict(params)

    # Take batch dimension from training experiment
    params['batch_dim'] = str2cls(f'{params["training_experiment"]}.get_batch_dim')()
    params = Bunch(params)

    set_seed(_seed)
    # run_deterministic()
    # Each worker process gets assigned two threads, so we can use them all in the main evaluation
    set_threads(params.threads_per_process * params.exp_processes)

    # Load the dataset with the params from the grid search experiment, i.e. the val set
    # Split dataset into validation and test set
    # Default Pipeline is taken from the training experiment
    pipeline = str2cls(f'{params.training_experiment}.get_test_pipeline')()
    if isinstance(dataset['pipeline'], collections.abc.Sequence):
        pipeline = [sacred.utils.recursive_update(copy.deepcopy(pipeline), pipe) for pipe in dataset['pipeline']]
    else:
        sacred.utils.recursive_update(pipeline, dataset['pipeline'])

    dataset             = remove_sacred_garbage(dataset)
    dataset['pipeline'] = pipeline

    training_param_grid = list(param_grid_to_list_of_dicts(training_param_grid))
    updated_param_grid  = [sacred.utils.recursive_update(copy.deepcopy(training_param_updates), point)
                           for point in training_param_grid]
    detector_param_grid = list(param_grid_to_list_of_dicts(detector_param_grid))

    # Note: We need to use spawn here, otherwise CUDA will produce errors
    context = torch.multiprocessing.get_context('spawn')

    if params.train_ids is None:

        # Only train if no information on already finished training runs is provided
        train_proc_func = functools.partial(train_once, params=params, seed=_seed)

        with Pool(max_workers=params.exp_processes, mp_context=context) as pool:
            train_ids = list(pool.map(train_proc_func, updated_param_grid))

    else:
        train_ids = params.train_ids

    final_scores = {metric: 0.0 for metric in params.evaluation_metrics}
    for val_fold in range(params.test_folds):
        # Split the dataset into folds. We use a padding of 1 fold around the validation fold to decrease the
        # probability of statistical dependencies between test and validation set
        splits, val_fold_index, test_fold_indices = compute_val_splits(params.test_folds, val_fold, padding=params.padding)
        dataset['split'] = splits

        _run.info[f'fold_{val_fold}'] = {}
        best_model, best_params, best_metric, best_scores, best_id, training_experiments, detector_files, all_val_scores = \
            evaluate_all(train_ids, updated_param_grid, detector_param_grid, params, dataset, val_fold_index, context, _seed)

        print(f'Best validation {params.validation_metric} was:', best_metric)
        print('Best parameters were:', best_params)
        _run.info[f'fold_{val_fold}']['best_params'] = best_params
        _run.info[f'fold_{val_fold}']['best_train_id'] = best_id
        _run.info[f'fold_{val_fold}']['best_val_scores'] = best_scores
        _run.info[f'fold_{val_fold}']['training_experiments'] = training_experiments
        _run.info[f'fold_{val_fold}']['all_val_scores'] = all_val_scores

        # Compute metrics for the best model on the test set
        test_evaluator = Evaluator()
        # load the best detector
        state = torch.load(best_model['detector'])
        for det_file in detector_files:
            try:
                os.remove(det_file)
            except FileNotFoundError:
                # We don't care if the file was already deleted
                pass

        detector = state['detector']
        del state
        detector = detector.to(params.device)
        detector.eval()
        best_model['detector'] = detector
        with make_experiment_tempfile(f'final_model_fold_{val_fold}.pth', _run, mode='wb') as f:
            torch.save(best_model, f)
        del best_model

        if 'dataset' in best_params and 'pipeline' in best_params['dataset']:
            sacred.utils.recursive_update(dataset['pipeline'], best_params['dataset']['pipeline'])
        if 'dataset' in best_params and 'use_dataset_pipeline' in best_params['dataset']:
            dataset['use_dataset_pipeline'] = best_params['dataset']['use_dataset_pipeline']

        data = load_dataset(**dataset)

        test_scores = defaultdict(list)
        for test_fold in test_fold_indices:
            test_ds = data[test_fold]
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=params.batch_size, num_workers=0,
                                                      collate_fn=collate_fn(params.batch_dim))

            labels, scores = detector.get_labels_and_scores(test_loader)
            for metric in params.evaluation_metrics:
                test_score, info = test_evaluator.__getattribute__(metric)(labels, scores)
                test_scores[metric].append(dict(score=test_score, info=info, test_fold=test_fold))
                final_scores[metric] += test_score / len(test_fold_indices) / params.test_folds

        _run.info[f'fold_{val_fold}'][f'test_scores'] = test_scores
        print(f'Evaluated test scores of that model (fold {val_fold}):', test_scores)

        # Clear cache
        del detector
        clear_gpu_memory()

    print('Final test scores:', final_scores)
    _run.info['final_scores'] = final_scores

    # Validate on the entire labelled data to get a single set of hyperparameters
    # Split the dataset into folds. We use a padding of 1 fold around the validation fold to decrease the
    # probability of statistical dependencies between test and validation set
    splits, val_fold_index, test_fold_indices = [1], 0, []
    dataset['split'] = splits

    best_model, best_params, best_metric, best_scores, best_id, training_experiments, detector_files, all_val_scores = \
        evaluate_all(train_ids, updated_param_grid, detector_param_grid, params, dataset, val_fold_index, context, _seed)

    _run.info['final_best_params'] = best_params
    _run.info['final_validation_scores'] = all_val_scores

    # I don't think we can just set the reference, so update the dict to make sure sacred gets it
    _run.info.update(remove_sacred_garbage(_run.info))

    return SerializationGuard(dict(test_scores=final_scores))
