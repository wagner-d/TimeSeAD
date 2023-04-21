import os
import random
import tempfile
from typing import Dict, Any, Union, Tuple, Optional, List
from unittest import TestCase

import torch

from timesead.data.dataset import BaseTSDataset
from timesead.data.tep_dataset import TEPDataset
from timesead.data.transforms import DatasetSource, PipelineDataset, WindowTransform, ReconstructionTargetTransform, \
    OneVsRestTargetTransform, SubsampleTransform
from timesead.utils.utils import ceil_div


class TensorDataset(BaseTSDataset):
    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self.tensor.shape[1]

    @property
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return self.tensor.shape[2]

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        pass

    def get_feature_names(self) -> List[str]:
        return [''] * self.num_features

    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, item):
        return (self.tensor[item],), ()

    def __len__(self):
        return len(self.tensor)


class TestTransform(TestCase):
    def test_mp_loading(self):
        data = torch.stack([torch.arange(20).unsqueeze(-1)] * 45, dim=0)
        dataset = PipelineDataset(DatasetSource(TensorDataset(data)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=5, num_workers=2)

        for i, (inputs, targets) in enumerate(loader):
            self.assertTrue(torch.all(data[i * 5:(i+1) * 5] == inputs[0]))

    def test_multiple_loops(self):
        time_steps = 20
        data_points = 45
        batch_size = 12

        data = torch.stack([torch.arange(time_steps).unsqueeze(-1)] * data_points, dim=0)
        dataset = PipelineDataset(DatasetSource(TensorDataset(data)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1)

        for i, (inputs, targets) in enumerate(loader):
            self.assertTrue(torch.all(data[i * batch_size: min((i + 1) * batch_size, data_points)] == inputs[0]))

        for i, (inputs, targets) in enumerate(loader):
            self.assertTrue(torch.all(data[i * batch_size: min((i + 1) * batch_size, data_points)] == inputs[0]))

    def test_save(self):
        data = torch.stack([torch.arange(20)] * 3, dim=1)
        dataset = PipelineDataset(DatasetSource(TensorDataset(data)))

        with tempfile.TemporaryDirectory() as dir:
            dataset.save(dir)

            load_data = torch.load(os.path.join(dir, 'data_0.pth'))
            self.assertTrue(torch.all(load_data[0][0] == data))

    def test_window(self):
        time_length = random.randint(10, 50)
        data = torch.stack([torch.arange(time_length).unsqueeze(-1)] * 20, dim=0)
        source = DatasetSource(TensorDataset(data))
        window_size = random.randint(1, time_length + 1)
        window = WindowTransform(source, window_size=window_size)
        dataset = PipelineDataset(window)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

        for i, (inputs, targets) in enumerate(loader):
            data_index, time_index = divmod(i, (time_length - window_size + 1))
            self.assertTrue(torch.all(data[data_index, time_index:time_index + window_size] == inputs[0][0]))

    def test_subsample(self):
        time_length = random.randint(10, 50)
        feature_dim = 1
        data = torch.arange(time_length*feature_dim).view(1, time_length, feature_dim)
        source = DatasetSource(TensorDataset(data))
        subsample_size = random.randint(1, time_length)
        subsample = SubsampleTransform(source, subsampling_factor=subsample_size)
        dataset = PipelineDataset(subsample)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

        for i, (inputs, targets) in enumerate(loader):
            new_t = ceil_div(time_length, subsample_size)
            true_label = torch.arange(start=0, end=new_t*subsample_size*feature_dim, step=subsample_size).view(new_t, feature_dim)
            self.assertTrue(torch.all(inputs[0] == true_label))

    def test_tep_processing(self):
        time_length = random.randint(10, 50)
        data = TEPDataset(training=False, faults=0)
        source = DatasetSource(data, axis='batch')
        window_size = random.randint(1, time_length)
        pipe = WindowTransform(source, window_size=window_size)
        pipe = ReconstructionTargetTransform(pipe, replace_labels=True)
        dataset = PipelineDataset(pipe)
        batch_size = random.randint(16, 129)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
        loader_iter = iter(loader)

        inputs, targets = next(loader_iter)
        self.assertEqual(inputs[0].shape, (batch_size, window_size, 52))
        self.assertEqual(targets[0].shape, (batch_size, window_size, 52))
        self.assertTrue(torch.all(inputs[0] == targets[0]))

    def test_onevsrest(self):
        data = TEPDataset(training=False, faults=1)
        source = DatasetSource(data)
        pipe = OneVsRestTargetTransform(source, normal_class=0, replace_labels=True)
        dataset = PipelineDataset(pipe)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)

        for i, (inputs, targets) in enumerate(loader):
            self.assertTrue(torch.max(targets[0]) <= 1)
            self.assertTrue(torch.min(targets[0]) >= 0)

    def test_negative_indices_in_datasource(self):
        time_length = random.randint(10, 50)
        feature_dim = 1
        num_ts = 25
        data = torch.arange(time_length * feature_dim).view(1, time_length, feature_dim)
        data = torch.cat([data] * num_ts, dim=0)

        start = random.randint(-num_ts, -1)
        end = start + random.randint(0, -start - 1)
        source = DatasetSource(TensorDataset(data), axis='batch', start=start, end=end)

        for i in range(end - start):
            self.assertTrue(torch.all(source.get_datapoint(i)[0][0] == data[start + i]))

    def test_negative_indices_in_datasource_time(self):
        time_length = random.randint(10, 50)
        feature_dim = 1
        num_ts = 25
        data = torch.arange(time_length * feature_dim).view(1, time_length, feature_dim)
        data = torch.cat([data] * num_ts, dim=0)

        start = random.randint(-time_length, -1)
        end = start + random.randint(0, -start - 1)
        source = DatasetSource(TensorDataset(data), axis='time', start=start, end=end)

        for i in range(end - start):
            self.assertTrue(torch.all(source.get_datapoint(i)[0][0] == data[i, start:end]))
