import random
from unittest import TestCase

import torch

from timesead.models.reconstruction import LSTMAE


class TestLSTMAE(TestCase):
    @staticmethod
    def get_random_params():
        seq_len = torch.randint(20, high=300, size=(1,)).item()
        batch_size = torch.randint(32, high=128, size=(1,)).item()
        input_dimension = torch.randint(20, high=100, size=(1,)).item()

        hidden_layers = torch.randint(input_dimension // 2, input_dimension * 2,
                                      size=(1,)).numpy().tolist()
        batch_first = random.random() < 0.5
        training = random.random() < 0.5

        kwargs = dict(input_dimension=input_dimension, hidden_dimensions=hidden_layers, batch_first=batch_first)
        return kwargs, seq_len, batch_size, training

    def random_forward(self):
        kwargs, seq_len, batch_size, training = self.get_random_params()
        input_dimension = kwargs['input_dimension']
        print('Parameters for test:', kwargs, "Training:", training)

        lstm_ae = LSTMAE(**kwargs)
        if training:
            lstm_ae.train()
        else:
            lstm_ae.eval()

        if kwargs['batch_first']:
            input = torch.rand(batch_size, seq_len, input_dimension)
        else:
            input = torch.rand(seq_len, batch_size, input_dimension)
        print('Input size:', input.shape)

        output = lstm_ae((input,))

        self.assertEqual(output.shape, input.shape)

    def test_forward(self):
        for _ in range(20):
            self.random_forward()