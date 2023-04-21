import logging
import os
from typing import List, Tuple, Dict, Union, Any

import torch

from ..dataset import BaseTSDataset
from .dataset_source import DatasetSource
from .transform_base import Transform
from ...utils.utils import objspec2constructor

_logger = logging.getLogger(__name__)


def make_pipe_from_dict(pipeline: Dict[str, Dict[str, Any]], data_source: DatasetSource) -> 'PipelineDataset':
    """
    Instantiates a :class:`PipelineDataset` from a given :class:`~timesead.data.transforms.DatasetSource` and a
    pipeline specification.

    .. warning::
        In case the specification of a :class:`~timesead.data.transforms.Transform` in the pipeline is incomplete or its
        instantiation fails for some other reason, this function simply prints a warning and continues with the next
        :class:`~timesead.data.transforms.Transform` instead of raising the exception.

    :param pipeline: Specification of the pipeline as a dict in the following format::

            {
                '<name>': {'class': '<name-of-transform-class>', 'args': {'<args-for-constructor>': <value>, ...}},
                ...
            }

        The function respects the order of transforms specified in the dict. That is, the first transform specified in
        the dict will be the first transform added to the pipeline and so on.
    :param data_source: The :class:`~timesead.data.transforms.DatasetSource` that acts as a source transform for the
        pipeline.
    :return: A :class:`PipelineDataset` that retrieves data from the given :class:`DatasetSource` and then executes the
        specified pipeline.
    """

    pipe = data_source
    for name, pipe_info in pipeline.items():
        try:
            pipe_class = objspec2constructor(pipe_info, base_module='timesead.data.transforms')
            pipe = pipe_class(pipe)
        except Exception:
            _logger.warning(f'Could not instantiate a Transform from the specification {pipe_info}! Ignoring.',
                            exc_info=True)
            continue

    return PipelineDataset(pipe)


class PipelineDataset(BaseTSDataset):
    """
    Dataset that can be used with a :class:`torch.utils.data.DataLoader` and executes a pipeline of transforms to
    retrieve its datapoints.
    """
    def __init__(self, sink_transform: Transform):
        """

        :param sink_transform: The last :class:`~timesead.data.transforms.Transform` in the pipeline that should be
            queried for data points.
        """
        super(PipelineDataset, self).__init__()

        self.sink_transform = sink_transform

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        return self.sink_transform.get_datapoint(item)

    def __len__(self):
        return len(self.sink_transform)

    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self.sink_transform.seq_len

    @property
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return self.sink_transform.num_features

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {}

    @staticmethod
    def __concatenate_and_save(out_inputs: Tuple[List[torch.Tensor], ...],
                               out_targets: Tuple[List[torch.Tensor], ...], file_name: str, batch_dim: int = 0):
        out_inputs = tuple(torch.stack(inp, dim=batch_dim) for inp in out_inputs)
        out_targets = tuple(torch.stack(tar, dim=batch_dim) for tar in out_targets)
        torch.save((out_inputs, out_targets), file_name)

    @staticmethod
    def get_feature_names() -> List[str]:
        raise NotImplementedError

    def save(self, path: str, chunk_size: int = 0, batch_dim: int = 0):
        """
        Save this dataset as it would be returned after all processing by its transforms is done.

        :param path: The folder in which to save the dataset.
        :param chunk_size: The maximum number of data points that should be saved in one file. If there are more data
            points than this value, multiple files will be created. Set this to 0 to save the entire dataset in one file.
        :param batch_dim: All (or `chunk_size`) datapoints will be stacked along this axis in a new tensor that is
           then saved to disk.
        """
        os.makedirs(path, exist_ok=True)

        out_inputs = None
        out_targets = None
        for i, (inputs, targets) in enumerate(self):
            if out_inputs is None:
                out_inputs = tuple([] for _ in inputs)
            for out_list, inp in zip(out_inputs, inputs):
                out_list.append(inp)

            if out_targets is None:
                out_targets = tuple([] for _ in targets)
            for out_list, target in zip(out_targets, targets):
                out_list.append(target)

            if chunk_size > 0 and i % chunk_size == chunk_size - 1:
                self.__concatenate_and_save(out_inputs, out_targets, os.path.join(path, f'data_{i // chunk_size}.pth'),
                                            batch_dim)
                out_inputs = out_targets = None

        # Save last chunk
        if len(out_inputs[0]) > 0:
            file_name = 'data_0.pth' if chunk_size == 0 else f'data_{i // chunk_size}.pth'
            self.__concatenate_and_save(out_inputs, out_targets, os.path.join(path, file_name), batch_dim)
