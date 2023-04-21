from unittest import TestCase

import torch

from timesead.models.reconstruction import TKNAutoencoder


class TestTKNAutoencoder(TestCase):
    def setUp(self) -> None:
        self.batch_size = 128
        self.window = 512
        self.num_features = 25
        self.kerv_ae = TKNAutoencoder(self.num_features)

    def test_forward(self):
        input = torch.rand((self.batch_size, self.window, self.num_features))
        reconstruction = self.kerv_ae((input,))
        self.assertEqual(input.shape, reconstruction.shape)
