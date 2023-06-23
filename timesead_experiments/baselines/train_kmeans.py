import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.baselines.kmeans import KMeansAD
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 12}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 50}}
    }


def get_batch_dim():
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True
    )

    split = (0.0, 1.0)


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict()

    detector_params = dict(
        k=20,
    )

    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():
    # Not training done, so returning None
    _, val_ds = load_dataset()
    return None, get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, detector_params, training, _run, save_detector=True):
    """
    We add a separate command to create an instance of AnomalyDetector from a model and return it. This is used
    in grid search so that we don't need to retrain the model when simply changing a parameter in the
    detector.
    """
    training = Bunch(training)
    detector = KMeansAD(batch_size=training['batch_size'], **detector_params).to(training.device)
    detector.fit(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):
    _, val_ds = get_datasets().object

    if train_detector:
        detector = get_anomaly_detector(None, val_ds)
    else:
        detector = None

    # Note that grid_search depends on you returning a dict that contains at least the model and the trainer.
    return dict(detector=detector, model=None)
