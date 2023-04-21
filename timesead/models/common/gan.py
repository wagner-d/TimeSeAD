from typing import Tuple

import torch

from ...optim.loss import Loss
from ...utils import torch_utils


class GAN(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        z, real_x = inputs
        # Generate x from z
        fake_x = self.generator(z)

        real_x_score = self.discriminator(real_x)
        fake_x_score = self.discriminator(fake_x)

        return fake_x, real_x_score, fake_x_score


class GANDiscriminatorLoss(Loss):
    def __init__(self):
        """
        This is the original GAN loss, i.e., - E[log(D(x))] - E[log(1 - D(G(z)))]
        """
        super(GANDiscriminatorLoss, self).__init__()

        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, fake_x_score = predictions

        loss_real = self.cross_entropy(real_x_score, torch.ones_like(real_x_score))
        loss_fake = self.cross_entropy(fake_x_score, torch.zeros_like(fake_x_score))

        return loss_real + loss_fake


class GANGeneratorLoss(Loss):
    def __init__(self):
        """
        This is the original GAN loss, i.e., E[log(1 - D(G(z)))]
        """
        super(GANGeneratorLoss, self).__init__()

        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, fake_x_score = predictions

        loss_fake = self.cross_entropy(fake_x_score, torch.zeros_like(fake_x_score))

        return -loss_fake


class GANGeneratorLossMod(Loss):
    def __init__(self):
        """
        This is a modified version of original GAN loss, i.e., -E[log(D(G(z)))]
        """
        super(GANGeneratorLossMod, self).__init__()

        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, fake_x_score = predictions

        loss_fake = self.cross_entropy(fake_x_score, torch.ones_like(fake_x_score))

        return loss_fake


def random_weighted_average(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    # Input shape is expected to be (B, *)
    alpha = torch.rand(input1.shape[0], dtype=input1.dtype, device=input1.device)
    # Add dimensions to alpha to make the broadcasting work
    alpha = alpha.view(-1, *([1]*(len(input1.shape) - 1)))
    return alpha * input1 + (1 - alpha) * input2


class WassersteinDiscriminatorLoss(Loss):
    def __init__(self, gan: GAN = None, gradient_penalty: float = 10):
        super(WassersteinDiscriminatorLoss, self).__init__()

        self.gradient_penalty_coeff = gradient_penalty
        self.gan = gan

    @staticmethod
    def gradient_penalty(discriminator, real_input: torch.Tensor, fake_input: torch.Tensor) -> torch.Tensor:
        interpolated = random_weighted_average(real_input, fake_input)
        interpolated_score = discriminator(interpolated)

        gradients = torch.autograd.grad(interpolated_score.sum(), interpolated, create_graph=True)[0]
        gradients = gradients.reshape(gradients.shape[0], -1)
        gradients_sqr_sum = torch_utils.batched_dot(gradients, gradients)
        gradient_l2_norm = torch.sqrt(gradients_sqr_sum)
        gradient_penalty = (1 - gradient_l2_norm)**2
        return torch.mean(gradient_penalty)

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, fake_x_score = predictions
        real_x, = targets

        wasserstein_loss = torch.mean(fake_x_score) - torch.mean(real_x_score)

        if self.gan is not None and self.gan.training and self.gradient_penalty_coeff != 0:
            wasserstein_loss = wasserstein_loss + self.gradient_penalty(self.gan.discriminator, real_x, fake_x)

        return wasserstein_loss


class WassersteinGeneratorLoss(Loss):
    def __init__(self):
        super(WassersteinGeneratorLoss, self).__init__()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake, real_score, fake_score = predictions

        return -torch.mean(fake_score)
