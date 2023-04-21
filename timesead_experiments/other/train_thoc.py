import torch

from timesead_experiments.utils import (
    data_ingredient,
    load_dataset,
    training_ingredient,
    train_model,
    make_experiment,
    make_experiment_tempfile,
    serialization_guard,
    get_dataloader,
)
from timesead.models.other import THOC, THOCAnomalyDetector, THOCLoss, THOCTrainer
from timesead.optim.trainer import EarlyStoppingHook
from timesead.utils.utils import Bunch
from timesead.data.dataset import collate_fn


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    """
    This method should return the updates to the pipeline that are specific to this method.
    Examples include the window size or a reconstruction target.

    This pipeline is used during training.

    :return: pipeline as a dict. This will be merged with the default dataset pipeline.
    """
    return {
        "window": {"class": "WindowTransform", "args": {"window_size": 64}},
        "reconstruction": {
            "class": "ReconstructionTargetTransform",
            "args": {"replace_labels": True},
        },
    }


def get_test_pipeline():
    """
    This method should return the updates to the pipeline that are specific to this method.
    Examples include the window size or a reconstruction target.

    This pipeline is used during testing, for example, by the grid_search experiment.

    :return: pipeline as a dict. This will be merged with the default dataset pipeline.
    """
    return {"window": {"class": "WindowTransform", "args": {"window_size": 64}}}


def get_batch_dim():
    """
    This method should return the dimension that should be used to concatenate data points into batches.
    """
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(training=True)

    split = (0.75, 0.25)


@training_ingredient.config
def training_config():
    trainer_hooks = [("post_validation", EarlyStoppingHook)]
    trainer = {
        "class": THOCTrainer,
        "args": dict(
            tau_decrease_steps=5, tau_decrease_gamma=2.0 / 3.0, init_centers_batches=20
        ),
    }
    batch_dim = get_batch_dim()
    optimizer = {
        "args": dict(
            lr=1e-3,
            weight_decay=1e-6,
        ),
    }
    scheduler = {
        "class": torch.optim.lr_scheduler.StepLR,
        "args": dict(
            step_size=20,
            gamma=0.65,
        )
    }


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        hidden_sizes=128,  # Size of the latent space
        n_hidden_layers=3,
        dilations=[1, 2, 4],
        clusters_dims=6,
        tau=100.0,
    )
    # Loss-specific parameters
    loss_params = dict(
        lambda_orth=1.0,
        lambda_tss=10.0
    )

    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():

    train_ds, val_ds = load_dataset()

    return get_dataloader(train_ds), get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard("model", "val_loader")
def get_anomaly_detector(model, val_loader, training, _run, save_detector=True):
    """
    We add a separate command to create an instance of AnomalyDetector from a model and return it. This is used
    in grid search so that we don't need to retrain the model when simply changing a parameter in the
    detector.
    """
    training = Bunch(training)
    detector = THOCAnomalyDetector(model).to(training.device)

    if save_detector:
        with make_experiment_tempfile("final_model.pth", _run, mode="wb") as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(
    model_params,
    loss_params,
    dataset,
    training,
    _run,
    train_detector=True,
):
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = THOC(train_ds.num_features, **model_params)
    loss = THOCLoss(model, **loss_params)

    trainer = train_model(_run, model, train_ds, val_ds, loss=[loss, loss])
    early_stop = trainer.hooks["post_validation"][-1]
    model = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    # Note that grid_search depends on you returning a dict that contains at least the model and the trainer.
    return dict(detector=detector, model=model)
