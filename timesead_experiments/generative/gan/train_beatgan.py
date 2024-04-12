import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.generative import BeatGANModel, BeatGANDiscriminatorLoss, BeatGANGeneratorLoss, \
    BeatGANReconstructionAnomalyDetector
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 320}},
        'augmentation': {'class': 'timesead.models.generative.WrapAugmentTransform', 'args': {'distort_fraction': 0.05,
                                                                                     'n_augmentations': 2}},
        'cache2': {'class': 'CacheTransform', 'args': {}},
        'reconstruction': {'class': 'ReconstructionTargetTransform', 'args': {'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 320}}
    }


def get_batch_dim():
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True
    )

    split = (1, 0)


@training_ingredient.config
def training_config():
    loss = [BeatGANDiscriminatorLoss, BeatGANGeneratorLoss]
    batch_dim = get_batch_dim()
    trainer_hooks = []
    scheduler = {
        'class': torch.optim.lr_scheduler.MultiStepLR,
        'args': dict(milestones=[20], gamma=0.1)
    }

@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        conv_filters=32,
        latent_dim=50,
    )

    loss_params = dict(
        adversarial_weight=1.0
    )

    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():
    # Validation dataset is not used in this experiment
    train_ds, _ = load_dataset()

    return get_dataloader(train_ds), None

@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, training, _run, save_detector=True):
    training = Bunch(training)
    detector = BeatGANReconstructionAnomalyDetector(model).to(training.device)
    # detector.fit(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, loss_params, dataset, training, _run, train_detector=True):
    model_params = Bunch(model_params)
    loss_params = Bunch(loss_params)
    ds_params = Bunch(dataset)
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = BeatGANModel(train_ds.num_features, model_params.conv_filters, model_params.latent_dim, last_kernel_size=train_ds.seq_len//32)

    gen_loss = BeatGANGeneratorLoss(loss_params.adversarial_weight)
    disc_loss = BeatGANDiscriminatorLoss()

    trainer = train_model(_run, model, train_ds, val_ds, loss=[disc_loss, gen_loss])
    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
