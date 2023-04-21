import torch

from timesead.models.prediction import TCNS2SPrediction, TCNS2SPredictionAnomalyDetector
from timesead.optim.trainer import EarlyStoppingHook
from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'prediction': {'class': 'OverlapPredictionTargetTransform', 'args': {'offset': 1, 'replace_labels': True}},
        'window': {'class': 'WindowTransform', 'args': {'window_size': 50}}
    }


def get_test_pipeline():
    return {
        'prediction': {'class': 'OverlapPredictionTargetTransform', 'args': {'offset': 1, 'replace_labels': False}},
        'window': {'class': 'WindowTransform', 'args': {'window_size': 50}}
    }


def get_batch_dim():
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True,
    )

    split = (0.75, 0.25)


@training_ingredient.config
def training_config():
    batch_dim = get_batch_dim()
    loss = torch.nn.MSELoss
    trainer_hooks = [
        ('post_validation', EarlyStoppingHook)
    ]
    scheduler = {
        'class': torch.optim.lr_scheduler.MultiStepLR,
        'args': dict(milestones=[20], gamma=0.1)
    }


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        filters=(64, 64, 64, 64, 64),
        kernel_sizes=(3, 3, 3, 3, 3),
        dilations=(1, 2, 4, 8, 16),
        last_n_layers_to_cat=3
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
def get_anomaly_detector(model, val_loader, training, dataset, _run, save_detector=True):
    training = Bunch(training)

    if val_loader is None:
        train_ds, val_ds = load_dataset()
        # Train for 0 epochs to get the val loader
        trainer = train_model(_run, model, train_ds, val_ds, epochs=0)
        val_loader = trainer.val_iter

    offset = dataset['pipeline']['prediction']['args']['offset']
    detector = TCNS2SPredictionAnomalyDetector(model, offset).to(training.device)
    detector.fit(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):
    ds_params = Bunch(dataset)
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = TCNS2SPrediction(train_ds.num_features, **model_params)

    trainer = train_model(_run, model, train_ds, val_ds)
    early_stop = trainer.hooks['post_validation'][-1]
    model = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
