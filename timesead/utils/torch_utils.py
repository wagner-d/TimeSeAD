import gc
import math
import os
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

activations = {
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh(),
    'linear': torch.nn.Identity()
}


def run_deterministic() -> None:
    # We need to set this environment variable to make CUDA use deterministic algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def run_fast() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def set_threads(n : Optional[int] = None) -> None:
    # Set to number of processors
    import torch.multiprocessing as mp
    cpu_count = mp.cpu_count()

    if n is None:
        n = cpu_count

    n = min(n, cpu_count)

    torch.set_num_threads(n)


def collate_fn_variable_length_ts(tensors : List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.nn.utils.rnn.PackedSequence, torch.nn.utils.rnn.PackedSequence]:
    return pack_sequence([t[0] for t in tensors], enforce_sorted=False), pack_sequence([t[1] for t in tensors], enforce_sorted=False)


def unpack_sequence(packed_sequence : torch.nn.utils.rnn.PackedSequence) -> List[torch.Tensor]:

    lengths = [0 for _ in range(len(packed_sequence.unsorted_indices))]
    for bs in packed_sequence.batch_sizes:
        for i in range(bs.data.numpy()):
            lengths[i] += 1

    feature_dimension = packed_sequence.data.shape[1:]

    unpacked_sequence = [torch.zeros(l, *feature_dimension) for l in lengths]

    head = 0

    for l_idx, b_size in enumerate(packed_sequence.batch_sizes):
        for b_idx in range(b_size):
            unpacked_sequence[b_idx][l_idx] = packed_sequence.data[head]
            head += 1

    return [unpacked_sequence[idx] for idx in packed_sequence.unsorted_indices]


def generate_random_sequences_like(sequences : Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence]) -> Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence]:

    if isinstance(sequences, torch.nn.utils.rnn.PackedSequence):

        unpacked_sequences = unpack_sequence(sequences)

        z = []
        for sequence in unpacked_sequences:
            z.append(torch.rand(sequence.shape))

        return pack_sequence(z, enforce_sorted=False)

    else:
        return torch.rand(sequences.shape)


def generate_random_sequences(shape : Union[List[Tuple[int, int]], Tuple[int, int]], n : int = 1) -> Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence]:

    if isinstance(shape, tuple):

        assert n > 0
        return torch.rand((n, *shape))

    else:

        sequences = []
        for s in shape:
            sequences.append(torch.rand(s))

        return pack_sequence(sequences, enforce_sorted=False)


def dataset_to_numpy_array(dataset : Union[torch.utils.data.Dataset, torch.utils.data.DataLoader], n : int = -1) -> List[np.array]:
    """
    Accumulates elements from a torch dataset into a numpy array.

    :param dataset: Dataset.
    :type dataset: torch.utils.data.dataset.Dataset or torch.utils.data.DataLoader
    :param n: Maximum number of samples collected from the dataset.
    :type n: int
    :return: Dataset as a list of numpy arrays.
    :rtype: list(numpy.array)
    """

    data_array = None

    if isinstance(dataset, torch.utils.data.DataLoader):

        for i, batch in enumerate(dataset):

            if data_array is None:

                if n > 0:
                    data_array = [batch[idx].detach().numpy()[:n] for idx in range(len(batch))]
                else:
                    data_array = [batch[idx].detach().numpy() for idx in range(len(batch))]

                continue

            if n > 0:
                data_array = [np.append(data_array[idx], batch[idx].detach().numpy()[:n-data_array.shape[0]], axis=0) for idx in range(len(batch))]
            else:
                data_array = [np.append(data_array[idx], batch[idx].detach().numpy(), axis=0) for idx in range(len(batch))]

            if data_array[0].shape[0] == n:
                break

    elif isinstance(dataset, torch.utils.data.Dataset):

        for i, batch in enumerate(dataset):

            if data_array is None:

                data_array = [np.expand_dims(batch[idx].detach().numpy(), axis=0) if isinstance(batch[idx], torch.Tensor) else np.expand_dims(batch[idx], axis=0) for idx in range(len(batch))]

                continue

            data_array = [np.append(data_array[idx], np.expand_dims(batch[idx].detach().numpy(), axis=0), axis=0) if isinstance(batch[idx], torch.Tensor) else np.append(data_array[idx], np.expand_dims(batch[idx], axis=0), axis=0) for idx in range(len(batch))]

            if data_array[0].shape[0] == n:
                break

    else:
        raise AssertionError('Expected Dataset or Dataloader object, but got {}.'.format(type(dataset)))

    return data_array


def sequences_to_dataframe(sequences : Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence]) -> List[pd.DataFrame]:

    if isinstance(sequences, torch.nn.utils.rnn.PackedSequence):

        unpacked_sequences = unpack_sequence(sequences)

        return [pd.DataFrame(sequence.detach().numpy()) for sequence in unpacked_sequences]

    else:

        np_sequences = sequences.detach().numpy()

        return [pd.DataFrame(np_sequences[sequence]) for sequence in range(np_sequences.shape[0])]


def list2tensor(input: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(input, torch.Tensor):
        return input

    return torch.stack(input)


def tensor2scalar(tensor: torch.Tensor) -> Union[torch.Tensor, float, int]:
    if tensor.numel() == 1:
        return tensor.item()

    return tensor


def batched_dot(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Computes a batched dot product.

    :param vec1: Tensor of shape (\*, D).
    :param vec2: Tensor of shape (\*, D). The batch shapes of both inputs must be broadcastable.
    :return: Tensor of shape (\*)
    """
    return torch.matmul(vec1.unsqueeze(-2), vec2.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def get_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def nested_list2tensor(inputs: List) -> torch.Tensor:
    # Get the dimensions of the final tensor.
    # Note that we assume that all tensors in the nested list have the same shape
    dimensions = []
    oerp = inputs
    while isinstance(oerp, list):
        dimensions.append(len(oerp))
        oerp = oerp[0]
    dimensions.extend(oerp.shape)

    def recursive_update(inp: Union[List, torch.Tensor], res: torch.Tensor):
        if isinstance(inp, list):
            for inp2, res2 in zip(inp, res):
                recursive_update(inp2, res2)
        else:
            res[...] = inp

    result = torch.empty(dimensions, dtype=oerp.dtype, device=oerp.device)
    recursive_update(inputs, result)

    return result


def exponential_moving_avg_(series: torch.Tensor, alpha: float, avg_num: float = 0, avg_denom: float = 0):
    """
    Computes the online adjusted exp weighted average of the errors.
    See https://pandas.pydata.org/docs/user_guide/window.html#window-exponentially-weighted.
    This modifies the given TS tensor in-place.
    """
    one_minus_alpha = 1 - alpha

    for t in range(len(series)):
        avg_num *= one_minus_alpha
        avg_num += series[t]
        avg_denom *= one_minus_alpha
        avg_denom += 1

        series[t] = avg_num / avg_denom

    return avg_num, avg_denom


def unsqueeze_like(to_squeeze : torch.Tensor, like : torch.Tensor, keep_dims : Optional[List] = None) -> torch.Tensor:

    if not keep_dims:
        return to_squeeze[(None,)*(len(like.shape) - len(to_squeeze.shape)) + (...,)]

    target = to_squeeze

    for i, (dim1, dim2) in enumerate(zip([-1] + keep_dims, keep_dims + [len(like.shape)])):
        for expands in range(dim2 - dim1 - 1):
            target = torch.unsqueeze(target, dim1+1)

    return target


def transform_tensor(tensor : Union[Tuple[torch.Tensor], torch.Tensor], current_shape : str, target_shape : str) -> Union[Tuple[torch.Tensor], torch.Tensor]:

    if isinstance(tensor, torch.Tensor):
        return tensor.permute(tuple(current_shape.index(i) for i in target_shape))
    else:
        return tuple(transform_tensor(t, current_shape, target_shape) for t in tensor)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.

    .. [Maechler2012accurate] Maechler, Martin (2012). Accurately Computing log(1-exp(-\|a\|)). Assessed from the Rmpfr package.
    """
    mask = -math.log(2) < x
    return torch.where(mask, (-x.expm1()).log(), (-x.exp()).log1p())


class ConstantLR(StepLR):
    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer, step_size=100, gamma=1)
