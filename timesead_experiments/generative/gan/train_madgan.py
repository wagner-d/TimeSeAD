import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.common import GANDiscriminatorLoss, GANGeneratorLossMod
from timesead.models.generative import MADGAN, MADGANTrainer, MADGANAnomalyDetector
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 120}},
        'reconstruction': {'class': 'ReconstructionTargetTransform', 'args': {'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 120}}
    }


def get_batch_dim():
    return 1


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True
    )

    split = (1, 0)


@training_ingredient.config
def training_config():
    loss = [GANDiscriminatorLoss, GANGeneratorLossMod]
    batch_dim = get_batch_dim()
    trainer = {
        'class': MADGANTrainer,
        'args': dict(
            disc_iterations=1,
            gen_iterations=3
        )
    }
    trainer_hooks = []
    scheduler = {
        'class': torch.optim.lr_scheduler.MultiStepLR,
        'args': dict(milestones=[20], gamma=0.1)
    }


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        latent_dim=15,
        generator_hidden_dims=[100, 100, 100],
        discriminator_hidden_dims=[100]
    )

    detector_params = dict(
        max_iter=1000,
        lambder=0.5,
        rec_error_tolerance=0.1
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
def get_anomaly_detector(model, val_loader, training, detector_params, _run, save_detector=True):
    training = Bunch(training)
    detector = MADGANAnomalyDetector(model, **detector_params).to(training.device)
    # detector.fit(val_loader)
    print(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):
    model_params = Bunch(model_params)
    ds_params = Bunch(dataset)
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = MADGAN(train_ds.num_features, **model_params._adict)

    trainer = train_model(_run, model, train_ds, val_ds)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
