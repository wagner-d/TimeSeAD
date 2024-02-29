import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.reconstruction import FEDformer
from timesead.models.common import MSEReconstructionAnomalyDetector
from timesead.optim.trainer import EarlyStoppingHook
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    """
    This method should return the updates to the pipeline that are specific to this method.
    Examples include the window size or a reconstruction target.

    This pipeline is used during training.

    :return: pipeline as a dict. This will be merged with the default dataset pipeline.
    """
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 100}},
        'reconstruction': {'class': 'ReconstructionTargetTransform', 'args': {'replace_labels': True}}
    }


def get_test_pipeline():
    """
    This method should return the updates to the pipeline that are specific to this method.
    Examples include the window size or a reconstruction target.

    This pipeline is used during testing, for example, by the grid_search experiment.

    :return: pipeline as a dict. This will be merged with the default dataset pipeline.
    """
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 100}}
    }


def get_batch_dim():
    """
    This method should return the dimension that should be used to concatenate data points into batches.
    """
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True
    )

    split = (0.75, 0.25)


@training_ingredient.config
def training_config():
    batch_dim = get_batch_dim()
    trainer_hooks = [
        ('post_validation', EarlyStoppingHook)
    ]


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        moving_avg=25,
        model_dim=128,
        dropout=0.1,
        num_heads=8,
        fcn_dim=128,
        activation='gelu',
        encoder_layers=3,
        version='fourier',
        mode_select='random',
        modes=32
    )

    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():

    train_ds, val_ds = load_dataset()

    return get_dataloader(train_ds), get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, training, _run, save_detector=True):
    """
    We add a separate command to create an instance of AnomalyDetector from a model and return it. This is used
    in grid search so that we don't need to retrain the model when simply changing a parameter in the
    detector.
    """
    training = Bunch(training)
    detector = MSEReconstructionAnomalyDetector(model).to(training.device)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):

    model_params = Bunch(model_params)
    ds_params    = Bunch(dataset)
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = FEDformer(
        train_ds.seq_len,
        train_ds.num_features,
        model_params.moving_avg,
        model_params.model_dim,
        model_params.dropout,
        model_params.num_heads,
        model_params.fcn_dim,
        model_params.activation,
        model_params.encoder_layers,
        version=model_params.version,
        mode_select=model_params.mode_select,
        modes=model_params.modes
    )

    trainer    = train_model(_run, model, train_ds, val_ds)
    early_stop = trainer.hooks['post_validation'][-1]
    model      = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    # Note that grid_search depends on you returning a dict that contains at least the model and the trainer.
    return dict(detector=detector, model=model)

