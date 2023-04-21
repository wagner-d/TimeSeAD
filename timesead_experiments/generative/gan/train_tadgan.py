import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.models.generative import TADGAN, TADGANGeneratorLoss, TADGANTrainer, TADGANAnomalyDetector
from timesead.models.common import WassersteinDiscriminatorLoss
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 100}},
        'reconstruction': {'class': 'ReconstructionTargetTransform', 'args': {'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 100}}
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
    batch_dim = get_batch_dim()
    optimizer = {
        'class': torch.optim.Adam,
        'args': dict(
            lr=0.0005
        )
    }

    trainer = {
        'class': TADGANTrainer,
        'args': dict(
            disc_iterations=5
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
        latent_size=20,
        enc_lstm_hidden_size=100,
        gen_lstm_hidden_size=64,
        disc_conv_filters=64,
        disc_conv_kernel_size=5,
        disc_z_hidden_size=20,
        gen_dropout=0.2,
        disc_dropout=0.25,
        disc_z_dropout=0.2
    )

    loss_params = dict(
        gradient_penalty=10,
        reconstruction_coeff=10
    )

    detector_params = dict(
        alpha=0.5
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
def get_anomaly_detector(model, val_loader, training, detector_params, loss_params, _run, save_detector=True):
    training = Bunch(training)
    loss_params = Bunch(loss_params)

    if val_loader is None:
        disc_z_loss = WassersteinDiscriminatorLoss(model.inverse_gan, loss_params.gradient_penalty)
        disc_x_loss = WassersteinDiscriminatorLoss(model.gan, loss_params.gradient_penalty)
        gen_loss = TADGANGeneratorLoss(loss_params.reconstruction_coeff)

        train_ds, val_ds = load_dataset()
        # Train for 0 epochs to get the val loader
        trainer = train_model(_run, model, train_ds, val_ds, loss=[disc_z_loss, disc_x_loss, gen_loss], epochs=0)
        val_loader = trainer.val_iter

    detector = TADGANAnomalyDetector(model, **detector_params).to(training.device)
    detector.fit(val_loader)

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
    model = TADGAN(train_ds.num_features, train_ds.seq_len, **model_params._adict)

    disc_z_loss = WassersteinDiscriminatorLoss(model.inverse_gan, loss_params.gradient_penalty)
    disc_x_loss = WassersteinDiscriminatorLoss(model.gan, loss_params.gradient_penalty)
    gen_loss = TADGANGeneratorLoss(loss_params.reconstruction_coeff)

    trainer = train_model(_run, model, train_ds, val_ds, loss=[disc_z_loss, disc_x_loss, gen_loss])

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
