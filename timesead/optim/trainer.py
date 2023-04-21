import functools
import itertools
import logging
import os
import time
from collections import defaultdict
from typing import Callable, Dict, Any, Union, Tuple, List, Optional

import torch
import tqdm

from .loss import Loss
from ..utils.torch_utils import tensor2scalar
from ..utils.utils import pack_tuple


SUPPORTED_HOOKS = ['post_validation']


class default_log_fn:
    def __init__(self):
        self.history = defaultdict(list)

    def __call__(self, metric_name: str, metric_value: Any):
        self.history[metric_name].append(metric_value)


class Trainer:
    def __init__(self, train_iter: torch.utils.data.DataLoader, val_iter: torch.utils.data.DataLoader,
                 optimizer: Callable = torch.optim.Adam,
                 scheduler: Callable = torch.optim.lr_scheduler.MultiStepLR,
                 device: Union[str, torch.device] = 'cpu', checkpoints: bool = False, out_dir: Optional[str] = None,
                 batch_dimension: int = 0):

        self.opt = optimizer
        self.sched = scheduler
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.device = device
        self.batch_dimension = batch_dimension

        # Hooks
        self.hooks = defaultdict(list)

        # Legacy interface for backward compatibility
        if checkpoints:
            self.add_hook(CheckpointHook(out_dir), 'post_validation')

    def validate_batch(self, network: torch.nn.Module, val_metrics: Dict[str, Callable],
                       b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> Dict[str, float]:
        res = pack_tuple(network(b_inputs))

        batch_metrics = {}
        for m_name, m in val_metrics.items():
            batch_metrics[m_name] = tensor2scalar(m(res, b_targets, b_inputs, *args, **kwargs).detach().cpu()) \
                                       * b_inputs[0].shape[self.batch_dimension]

        return batch_metrics

    def validate_model_once(self, network: torch.nn.Module, val_metrics: Dict[str, Callable], *args,
                            print_progress: bool = False, **kwargs) -> Dict[str, Any]:
        network.eval()
        total = 0
        current_metrics = {k: 0 for k in val_metrics.keys()}
        if print_progress:
            val_iter = tqdm.tqdm(self.val_iter, desc=f'Loss: {0:.3f}')
        else:
            val_iter = self.val_iter

        with torch.no_grad():
            for b_inputs, b_targets in val_iter:
                b_inputs = tuple(b_inp.to(self.device) for b_inp in b_inputs)
                b_targets = tuple(b_tar.to(self.device) for b_tar in b_targets)

                total += b_inputs[0].shape[self.batch_dimension]
                batch_metrics = self.validate_batch(network, val_metrics, b_inputs, b_targets, *args, **kwargs)
                current_metrics = {k: v_c + v_b for (k, v_c), (k, v_b)
                                   in zip(current_metrics.items(), batch_metrics.items())}

                if print_progress:
                    val_iter.set_description(
                        ', '.join(f'{m_name}: {m_value / total:.3f}' for m_name, m_value in current_metrics.items())
                    )

        for m_name in val_metrics:
            current_metrics[m_name] /= max(total, 1)

        return current_metrics

    def train_batch(self, network: torch.nn.Module, losses: List[Loss], optimizers: List[torch.optim.Optimizer],
                    epoch: int, num_epochs: int, b_inputs: Tuple[torch.Tensor, ...],
                    b_targets: Tuple[torch.Tensor, ...]) -> List[float]:
        batch_loss = [0 for _ in optimizers]
        for i, (optimizer, loss) in enumerate(zip(optimizers, losses)):
            res = pack_tuple(network(b_inputs))
            l = loss(res, b_targets, epoch=epoch, num_epochs=num_epochs)
            optimizer.zero_grad(True)
            opt_params = list(itertools.chain(*(group['params'] for group in optimizer.param_groups)))
            l.backward(inputs=opt_params)
            optimizer.step()
            optimizer.zero_grad(True)
            batch_loss[i] = tensor2scalar(l.detach().cpu()) * b_inputs[0].shape[self.batch_dimension]

        return batch_loss

    def train_epoch(self, network: torch.nn.Module, losses: List[Loss], optimizers: List[torch.optim.Optimizer],
                    schedulers: List[torch.optim.lr_scheduler._LRScheduler], epoch: int, num_epochs: int,
                    val_metrics: Dict[str, Callable], log_fn: Callable[[str, Any], None] = default_log_fn) -> bool:
        print('Training:', flush=True)
        network.train()
        train_iter = tqdm.tqdm(self.train_iter, desc=f'Loss: {0:.3f}')
        total, avg_loss = 0, [0 for _ in optimizers]
        start = time.perf_counter()
        for b_inputs, b_targets in train_iter:
            b_inputs = tuple(b_inp.to(self.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.device) for b_tar in b_targets)

            batch_loss = self.train_batch(network, losses, optimizers, epoch, num_epochs, b_inputs, b_targets)
            avg_loss = [a + b for a, b in zip(avg_loss, batch_loss)]

            total += b_inputs[0].shape[self.batch_dimension]
            train_iter.set_description(', '.join(f'Loss {i:d}: {loss / total:.3f}' for i, loss in enumerate(avg_loss)))

        end = time.perf_counter()
        log_fn('train_time', end - start)

        for i, l in enumerate(avg_loss):
            log_fn(f'train_loss_{i}', l / total)

        for scheduler in schedulers:
            scheduler.step()

        print('Validation:', flush=True)
        start = time.perf_counter()
        current_metrics = self.validate_model_once(network, val_metrics, print_progress=True, epoch=epoch,
                                                   num_epochs=num_epochs)
        end = time.perf_counter()
        log_fn('val_time', end - start)

        for m_name, m_value in current_metrics.items():
            log_fn(f'val_{m_name}', m_value)

        # Run post-validation hooks
        if not all(hook(self, network, optimizers, epoch, current_metrics) for hook in self.hooks['post_validation']):
            # Stop Training due to negative hook feedback
            return False

        return True

    def train(self, network: torch.nn.Module, losses: Union[List[Loss], Loss], num_epochs: int, val_metrics: Dict[str, Callable] = None,
              start_epoch: int = 0, log_fn: Callable[[str, Any], None] = default_log_fn):
        # We expect this to return a tuple of size k, where k is the number of parameter groups.
        # Each group gets its own optimizer and the loss function is expected to return the
        # same number of individual losses
        network = network.to(self.device)
        parameters = pack_tuple(network.grouped_parameters())
        if isinstance(losses, Loss):
            losses = [losses] * len(parameters)
        else:
            assert len(losses) == len(parameters)
        losses = [loss.to(self.device) for loss in losses]

        if val_metrics is None:
            val_metrics = {f'loss_{i}': loss for i, loss in enumerate(losses)}

        optimizers = [self.opt(params) for params in parameters]
        schedulers = [self.sched(optimizer) for optimizer in optimizers]

        for e in range(start_epoch, start_epoch + num_epochs):
            logging.info(f'Epoch {e + 1:d} of {start_epoch + num_epochs:d}:')
            if not self.train_epoch(network, losses, optimizers, schedulers, e, num_epochs, val_metrics, log_fn):
                break

    def add_hook(self, hook: Callable, type: str = 'post_validation'):
        if type not in SUPPORTED_HOOKS:
            raise ValueError(f'Hook type "{type}" is not supported!')

        self.hooks[type].append(hook)


def open_file_in_dir(name, out_dir, mode='w+b'):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    return open(os.path.join(out_dir, name), mode=mode)


class CheckpointHook:
    def __init__(self, out_dir: Optional[str] = None, checkpoint_interval: int = 10,
                 file_write_fn: Optional[Callable] = None, file_read_fn: Optional[Callable] = None):
        self.checkpoint_interval = checkpoint_interval

        if file_write_fn is None:
            if out_dir is None:
                raise ValueError('Either file_write_fn or out_dir must be specified!')
            file_write_fn = functools.partial(open_file_in_dir, out_dir=out_dir)
        self.file_write_fn = file_write_fn

        if file_read_fn is None:
            if out_dir is None:
                raise ValueError('Either file_read_fn or out_dir must be specified!')
            file_read_fn = functools.partial(open_file_in_dir, out_dir=out_dir)
        self.file_read_fn = file_read_fn

    def save_model(self, filename: str, network: torch.nn.Module, optimizers: List[torch.optim.Optimizer]):
        with self.file_write_fn(filename, mode='wb') as f:
            torch.save(dict(model=network.state_dict(), opt=[opt.state_dict() for opt in optimizers]), f)

    def load_model_state(self, filename: str,) -> Dict:
        with self.file_read_fn(filename, mode='rb') as f:
            state = torch.load(f)

        return state

    def __call__(self, trainer: Trainer, network: torch.nn.Module, optimizers: List[torch.optim.Optimizer],
                 epoch: int, val_metrics: Dict[str, float]) -> bool:
        if (epoch + 1) % self.checkpoint_interval != 0:
            return True

        self.save_model(f'snap_{epoch + 1:03d}.pth', network, optimizers)

        return True


class EarlyStoppingHook:
    def __init__(self, metric: str = 'loss_0', invert_metric: bool = True, patience: int = 10, epsilon: float = 0):
        self.metric = metric
        self.invert_metric = invert_metric
        self.patience = patience
        self.epsilon = epsilon

        self.__best_metric = -float('inf')
        self.best_epoch = 0
        self.decrease_counter = 0

    @property
    def best_metric(self) -> float:
        return -self.__best_metric if self.invert_metric else self.__best_metric

    def save_best_model(self, trainer: Trainer, network: torch.nn.Module, optimizers: List[torch.optim.Optimizer]):
        for hook in trainer.hooks['post_validation']:
            if isinstance(hook, CheckpointHook):
                hook.save_model('best_model.pth', network, optimizers)
                return

    def load_best_model(self, trainer: Trainer, network: torch.nn.Module, total_epochs: int):
        if total_epochs > 0 and self.best_epoch != total_epochs - 1:
            for hook in trainer.hooks['post_validation']:
                if isinstance(hook, CheckpointHook):
                    # Load model state from best epoch
                    state = hook.load_model_state('best_model.pth')
                    network.load_state_dict(state['model'])
                    del state
                    break

        return network

    def __call__(self, trainer: Trainer, network: torch.nn.Module, optimizers: List[torch.optim.Optimizer],
                 epoch: int, val_metrics: Dict[str, float]) -> bool:
        score = -val_metrics[self.metric] if self.invert_metric else val_metrics[self.metric]

        if score > self.__best_metric + self.epsilon:
            self.__best_metric = score
            self.best_epoch = epoch
            self.decrease_counter = 0
            self.save_best_model(trainer, network, optimizers)
        else:
            self.decrease_counter += 1
            if self.decrease_counter >= self.patience:
                return False

        return True

    def get_best_epoch(self, _print: bool = False) -> int:
        if _print:
            logging.info(f'Best Epoch was {self.best_epoch + 1:d} with {self.best_metric:.3f} average {self.metric}.')
        return self.best_epoch
