import collections.abc
import functools
from typing import Type, Union

import torch
import torch.utils.data
from sacred import Ingredient

from .experiment_functions import make_experiment_tempfile, open_artifact_from_run
from timesead.data.dataset import collate_fn
from timesead.optim.loss import TorchLossWrapper, Loss
from timesead.optim.trainer import Trainer, CheckpointHook
from timesead.utils.torch_utils import run_deterministic, run_fast, ConstantLR
from timesead.utils.utils import objspec2constructor
from timesead.utils.rng_utils import set_seed


training_ingredient = Ingredient('training')


@training_ingredient.config
def config():
    # Specify class and arguments to use for the optimizer
    optimizer = {
        'class': torch.optim.Adam,
        'args': dict()
    }
    # Batch size to use during training
    batch_size = 128
    # Dimension along which the batch tensor is stacked
    batch_dim = 0
    # Number of workers to load data. 0 means that all work is done in the main process
    num_workers = 0
    # Whether to remove the last batch of an epoch if it is incomplete, i.e., has fewer points than specified by
    # batch_size
    drop_last = False
    # Class and arguments to use for the LR scheduler
    scheduler = {
        'class': ConstantLR,
        'args': dict()
    }
    # Number of epochs to train for
    epochs = 70
    # The device to use for all computations. Can be 'cpu' or 'cuda[:<n>]' where n specifies the index of the device
    # if multiple NVIDIA GPUs are available
    device = 'cpu'
    # The loss class to use during training. Can also be a list of different loss classes, if you need more than one
    # loss. This also supports a dictionary format similar to the scheduler and optimizer
    loss = torch.nn.MSELoss
    # The trainer class to use. This also supports a dictionary format similar to the scheduler and optimizer
    trainer = Trainer
    # Hooks to use during training. A list of tuples. First element should be the event, second an objspec.
    trainer_hooks = [
        # ('post_validation', EarlyStoppingHook)
    ]
    # Number of epochs between checkpoints
    checkpoint_interval = 15
    # Whether to try and use deterministic algorithms wherever possible. Note that this may degrade performance
    # significantly and can even lead to errors on specific system configurations
    deterministic = False


def instantiate_loss(loss: Union[str, Loss, Type[Loss], torch.nn.modules.loss._Loss, Type[torch.nn.modules.loss._Loss]]) \
        -> Loss:
    if isinstance(loss, Loss):
        return loss
    if isinstance(loss, torch.nn.modules.loss._Loss):
        return TorchLossWrapper(loss)

    # It was not a loss object, so we have to instantiate it
    loss = objspec2constructor(loss)()

    if not isinstance(loss, Loss):
        # Wrap the standard pytorch classes to support multiple inputs
        loss = TorchLossWrapper(loss)

    return loss


@training_ingredient.capture
def train_model(_run, model, train_ds, val_ds, optimizer, batch_size, batch_dim, num_workers, scheduler, epochs, device,
                loss, trainer, trainer_hooks, checkpoint_interval, deterministic, drop_last, _seed):
    set_seed(_seed)

    if deterministic:
        run_deterministic()
    else:
        run_fast()

    # Load training data
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                               collate_fn=collate_fn(batch_dim), drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                             collate_fn=collate_fn(batch_dim), drop_last=drop_last)

    optimizer = objspec2constructor(optimizer)
    scheduler = objspec2constructor(scheduler)
    trainer = objspec2constructor(trainer)

    trainer = trainer(train_loader, val_loader, optimizer, scheduler, device=device, checkpoints=False,
                      batch_dimension=batch_dim)

    # Checkpoint after every epoch
    checkpoints = CheckpointHook(checkpoint_interval=checkpoint_interval,
                                 file_write_fn=functools.partial(make_experiment_tempfile, run=_run),
                                 file_read_fn=functools.partial(open_artifact_from_run, run=_run))
    trainer.add_hook(checkpoints, 'post_validation')

    # Add additional hooks
    for event, hook in trainer_hooks:
        trainer.add_hook(objspec2constructor(hook)(), event)

    if not isinstance(loss, collections.abc.Sequence):
        loss = [loss]

    # Create the loss objects
    loss = [instantiate_loss(l) for l in loss]

    trainer.train(model, loss, epochs, log_fn=_run.log_scalar)

    return trainer


@training_ingredient.capture
def get_dataloader(dataset, batch_size, num_workers, batch_dim, drop_last):

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                       collate_fn=collate_fn(batch_dim), drop_last=drop_last)

