import collections
import copy
import json
import logging
import os

import sacred
import torch
from sacred.serializer import restore

from timesead_experiments.utils import make_experiment, load_dataset, make_experiment_tempfile, remove_sacred_garbage
from timesead.data.dataset import collate_fn
from timesead.data.transforms import WindowTransform
from timesead.utils.metadata import LOG_DIRECTORY
from timesead.utils.utils import str2cls


experiment = make_experiment()


def get_model_predictions(path, fold=0, device=None):
    info_file = os.path.join(path, 'info.json')
    with open(info_file, mode='r') as f:
        info = json.load(f)
    # Because of sacred and jsonpickle we need to remove garbage twice!
    info = remove_sacred_garbage(info)
    info = restore(info)
    info = remove_sacred_garbage(info)

    config_file = os.path.join(path, 'config.json')
    with open(config_file, mode='r') as f:
        config = json.load(f)
    config = restore(config)

    # load the best detector
    exp_name = config['params']['training_experiment']
    exp_name_short = exp_name.rsplit('.', 1)[1]
    if fold < 0:
        file_name = 'final_model.pth'
    else:
        file_name = f'final_model_fold_{fold}.pth'
    detector = torch.load(os.path.join(path, file_name), map_location=device)
    detector = detector['detector']

    # Compute metrics for the best model on the test set
    dataset = config['dataset']
    pipeline = str2cls(f'{exp_name}.get_test_pipeline')()
    if isinstance(dataset['pipeline'], collections.abc.Sequence):
        pipeline = [sacred.utils.recursive_update(copy.deepcopy(pipeline), pipe) for pipe in dataset['pipeline']]
    else:
        sacred.utils.recursive_update(pipeline, dataset['pipeline'])
    dataset['pipeline'] = pipeline

    if fold < 0:
        best_params = info['best_params']
    else:
        best_params = info[f'fold_{fold}']['best_params']
    if 'dataset' in best_params:
        sacred.utils.recursive_update(dataset, best_params['dataset'])
    dataset['split'] = (1,)

    test_ds, = load_dataset(**dataset, _log=logging.getLogger())
    batch_dim = str2cls(f'{exp_name}.get_batch_dim')()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config['params']['batch_size'], num_workers=0,
                                             collate_fn=collate_fn(batch_dim))

    device = config['params']['device'] if device is None else device
    detector = detector.to(device)
    detector.eval()

    labels, scores = detector.get_labels_and_scores(test_loader)
    labels = labels.cpu()
    scores = scores.cpu()

    data = []
    for i, (inputs, targets) in enumerate(test_loader):
        x = inputs[0]

        last_step = x[-1] if batch_dim != 0 else x[:,-1]
        data.append(last_step)

    data = torch.cat(data, dim=0).cpu()

    # Get window transform
    transform = test_ds.sink_transform
    window_transform = None
    while transform is not None:
        if isinstance(transform, WindowTransform):
            window_transform = transform
            break

        transform = transform.parent

    if window_transform is None:
        # Fallback
        inverse_transform = lambda i: (0, 0)
    else:
        inverse_transform = lambda i: window_transform.inverse_transform_index(window_transform.window_size - 1 + i)

    # Split long concatenation along time series
    scores_list = collections.defaultdict(list)
    labels_list = collections.defaultdict(list)
    data_list = collections.defaultdict(list)
    for i, (score, label, dat) in enumerate(zip(scores, labels, data)):
        ts_index, _ = inverse_transform(i)
        scores_list[ts_index].append(score)
        labels_list[ts_index].append(label)
        data_list[ts_index].append(dat)

    scores_list = [torch.stack(scores_list[i]) for i in range(len(scores_list))]
    labels_list = [torch.stack(labels_list[i]) for i in range(len(labels_list))]
    data_list = [torch.stack(data_list[i]) for i in range(len(data_list))]

    return data_list, labels_list, scores_list, exp_name_short


@experiment.config
def config():
    # Model-specific parameters
    log_dir = LOG_DIRECTORY
    experiments = None
    fold = 0
    device = None


@experiment.automain
def main(log_dir, experiments, fold, device, _run):
    log_dir = os.path.join(log_dir, 'grid_search')

    if experiments is None:
        experiments = [entry for entry in os.scandir(log_dir) if not entry.startswith('_')]

    _run.info['exp_name'] = {}
    for experiment in experiments:
        data, labels, scores, exp_name = get_model_predictions(os.path.join(log_dir, str(experiment)),
                                                     fold=fold, device=device)

        _run.info['exp_name'][experiment] = exp_name

        with make_experiment_tempfile(f'{exp_name}_predictions.pth', _run, mode='wb') as f:
            torch.save(dict(data=data, labels=labels, scores=scores), f)
