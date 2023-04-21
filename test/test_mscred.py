from unittest import TestCase

import torch

from timesead.models.reconstruction.mscred import MSCRED, compute_signature_matrix


class TestMSCRED(TestCase):
    def setUp(self) -> None:
        self.batch_size = 128
        self.h = 5
        self.seg_interval = 10
        self.wins = (10, 30, 60)
        self.window = self.seg_interval * (self.h - 1) + max(self.wins)
        self.num_features = 25
        self.mscred = MSCRED(self.num_features, len(self.wins))

    def test_forward(self):
        input = torch.rand((self.batch_size, self.window, self.num_features))
        matrices = torch.stack([compute_signature_matrix(inp, self.seg_interval, self.wins, self.h) for inp in input])
        reconstruction = self.mscred((matrices,))
        self.assertEqual(matrices[-1].shape, reconstruction.shape)
