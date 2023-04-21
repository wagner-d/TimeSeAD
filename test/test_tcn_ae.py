import random
from unittest import TestCase

import torch

from timesead.models.reconstruction import TCNAE


class TestTCNAE(TestCase):
    @staticmethod
    def get_random_params():
        seq_len = torch.randint(20, high=300, size=(1,)).item()
        batch_size = torch.randint(32, high=128, size=(1,)).item()
        input_dimension = torch.randint(20, high=100, size=(1,)).item()

        batch_first = random.random() < 0.5
        training = random.random() < 0.5

        kwargs = dict(
            nb_filters = torch.randint(10, high=30, size=(1,)).item(),
            kernel_size = torch.randint(1, high=seq_len//3, size=(1,)).item(),
            nb_stacks= 1,
            dropout_rate = 0.00,
            filters_conv1d = 8,
            latent_sample_rate = torch.randint(1, high=seq_len//3, size=(1,)).item(),
        )

        kwargs.update(input_dimension=input_dimension, batch_first=batch_first)
        return kwargs, seq_len, batch_size, training

    def random_forward(self):
        kwargs, seq_len, batch_size, training = self.get_random_params()
        input_dimension = kwargs['input_dimension']
        print('Parameters for test:', kwargs, "Training:", training)

        lstm_ae = TCNAE(**kwargs)
        if training:
            lstm_ae.train()
        else:
            lstm_ae.eval()

        if kwargs['batch_first']:
            input = torch.rand(batch_size, seq_len, input_dimension)
        else:
            input = torch.rand(seq_len, batch_size, input_dimension)
        print('Input size:', input.shape)

        output = lstm_ae(input)

        self.assertEqual(output.shape, input.shape)

    def test_forward(self):
        for _ in range(20):
            self.random_forward()