import collections.abc
import logging
from typing import Dict, Any, Tuple, List

import sacred.utils
from sacred import Ingredient

from timesead.data.transforms import PipelineDataset, make_dataset_split, make_pipe_from_dict
from timesead.utils.utils import objspec2constructor


data_ingredient = Ingredient('dataset')


@data_ingredient.config
def data_config():
    # The dataset class to load. This should either be a fully-qualified name or relative to timesead.data
    name = 'SWaTDataset'
    # Arguments to pass to the dataset constructor
    ds_args = dict(
        training=True
    )

    # Pipeline to use for processing the data. See train_model_template.py for an example
    # This can also be a list of dicts, one for each split
    pipeline = {}
    # Setting this will apply the default pipeline for the dataset before the specified pipeline. Otherwise,
    # the dataset pipeline is overwritten by the pipeline specified here
    use_dataset_pipeline = True

    # Allows to split the dataset into several parts. Each number will be divided by the sum of all numbers and
    # interpreted as a percentage
    split = (0.75, 0.25)
    # Decides along which axis to split the data. Possible options are 'batch' and 'time'
    split_axis = 'time'


@data_ingredient.capture
def load_dataset(name: str, ds_args: Dict[str, Any], pipeline: Dict, use_dataset_pipeline: bool,
                 split: Tuple[float, ...], split_axis: str, _log: logging.Logger) -> List[PipelineDataset]:
    ds = objspec2constructor({'class': name, 'args': ds_args}, base_module='timesead.data')()

    if not isinstance(pipeline, collections.abc.Sequence):
        pipelines = [pipeline] * len(split)
    else:
        pipelines = pipeline

    assert len(pipelines) == len(split)

    ds_splits = make_dataset_split(ds, *split, axis=split_axis)
    res_splits = []
    for pipeline, ds_pipe in zip(pipelines, ds_splits):
        if use_dataset_pipeline:
            default_pipe = dict(ds.get_default_pipeline())
            sacred.utils.recursive_update(default_pipe, pipeline)
            pipeline = default_pipe

        pipe = make_pipe_from_dict(pipeline, ds_pipe)
        res_splits.append(pipe)

    return res_splits
