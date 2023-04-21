import random
from unittest import TestCase

import torch

from timesead.models.common import RNN


class TestRNN(TestCase):
    @staticmethod
    def get_random_params():
        seq_len = torch.randint(20, high=300, size=(1,)).item()
        batch_size = torch.randint(32, high=128, size=(1,)).item()
        input_dimension = torch.randint(20, high=100, size=(1,)).item()

        layer_type = random.choice(['lstm', 'gru', 'rnn'])
        n_hidden_layers = torch.randint(1, high=11, size=(1,)).item()
        hidden_layers = torch.randint(input_dimension // 2, input_dimension * 2,
                                      size=(n_hidden_layers,)).numpy().tolist()
        dilation = [2 ** i for i in range(n_hidden_layers)] if random.random() < 0.5 else None
        batch_first = random.random() < 0.5

        kwargs = dict(layer_type=layer_type, input_dimension=input_dimension,
                      hidden_dimensions=hidden_layers, batch_first=batch_first, dilation=dilation)
        return kwargs, seq_len, batch_size

    def random_forward_s2s(self):
        kwargs, seq_len, batch_size = self.get_random_params()
        input_dimension = kwargs['input_dimension']
        kwargs['batch_first'] = False
        kwargs['model'] = 's2s'
        print('Parameters for test:', kwargs)

        rnn = RNN(**kwargs)

        input = torch.rand(seq_len, batch_size, input_dimension)
        print('Input size:', input.shape)

        output = rnn(input)

        self.assertEqual(output.shape, (seq_len, batch_size, kwargs['hidden_dimensions'][-1]))

    def test_forward_s2s(self):
        for _ in range(20):
            self.random_forward_s2s()

    def random_forward_s2fh(self):
        kwargs, seq_len, batch_size = self.get_random_params()
        kwargs['model'] = 's2fh'
        print('Parameters for test:', kwargs)

        rnn = RNN(**kwargs)

        input = torch.rand(seq_len, batch_size, kwargs['input_dimension'])
        if kwargs['batch_first']:
            input = input.transpose(0, 1)
        print('Input size:', input.shape)

        output = rnn(input)

        self.assertEqual(output.shape, (batch_size, kwargs['hidden_dimensions'][-1]))

    def test_forward_s2fh(self):
        for _ in range(20):
            self.random_forward_s2fh()
