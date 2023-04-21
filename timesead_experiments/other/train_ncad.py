import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.other import NCADTrainer, NCADAnomalyDetector, NCAD
from timesead.optim.trainer import EarlyStoppingHook
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'outlier_injection': {'class': 'timesead.models.ncad.LocalOutlierInjectionTransform', 'args': dict(
            max_duration_spike=2,
            spike_multiplier_range=(0.5, 2.0),
            spike_value_range=(-float('inf'), float('inf')),
            area_radius=100,
            num_spikes=10
        )},
        'window': {'class': 'WindowTransform', 'args': {'window_size': 500}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 500}}
    }


def get_batch_dim():
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
    loss = torch.nn.BCEWithLogitsLoss()
    batch_dim = get_batch_dim()
    trainer = {
        'class': NCADTrainer,
        'args': dict(
            coe_rate=0.5,
            mixup_rate=2.0
        )
    }
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
        suspect_window_length=10,
        tcn_kernel_size=7,
        tcn_layers=8,
        tcn_out_channels=16,
        tcn_maxpool_out_channels=32,
        embedding_rep_dim=64,
        normalize_embedding=True,
        distance='l2_distance',
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
    training = Bunch(training)
    detector = NCADAnomalyDetector(model).to(training.device)
    # detector.fit(val_loader)

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
    model = NCAD(train_ds.num_features, **model_params)

    trainer = train_model(_run, model, train_ds, val_ds)
    early_stop = trainer.hooks['post_validation'][-1]
    model = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
