from unittest import TestCase

import torch

from timesead.models.other.mtad_gat import GAT, MTAD_GAT, MTAD_GATLoss


class TestGAT(TestCase):
    def setUp(self) -> None:
        self.batch_size = 64
        self.num_nodes = 100
        self.num_features = 25
        self.gat = GAT(self.num_nodes, self.num_features)

    def test_forward(self):
        input = torch.rand((self.batch_size, self.num_nodes, self.num_features))
        res = self.gat(input)
        self.assertEqual(input.shape, res.shape)


class TestMTAD_GAT(TestCase):
    def setUp(self) -> None:
        self.batch_size = 128
        self.window = 100
        self.num_features = 25
        self.vae_hidden = 300
        self.gat = MTAD_GAT(self.num_features, self.window, vae_hidden_dim=self.vae_hidden)
        self.loss = MTAD_GATLoss(reduction='mean')

    def test_forward(self):
        input = torch.rand((self.batch_size, self.window, self.num_features))
        mlp_prediction, vae_z_mean, vae_z_std, vae_x_mean, vae_x_std = self.gat((input,))
        self.assertEqual(mlp_prediction.shape, torch.Size((self.batch_size, 1, self.num_features)))
        self.assertEqual(vae_z_mean.shape, torch.Size((self.batch_size, self.vae_hidden)))
        self.assertEqual(vae_z_std.shape, torch.Size((self.batch_size, self.vae_hidden)))
        self.assertEqual(vae_x_mean.shape, torch.Size((self.batch_size, self.window, self.num_features)))
        self.assertEqual(vae_x_std.shape, torch.Size((self.batch_size, self.window, self.num_features)))

        vae_loss = self.loss((mlp_prediction, vae_z_mean, vae_z_std, vae_x_mean, vae_x_std), (input[:,-1:], input))
        self.assertEqual(vae_loss.shape, torch.Size([]))
