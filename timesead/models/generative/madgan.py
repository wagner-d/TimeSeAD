from typing import List, Tuple, Iterator, Dict, Callable

import torch
import torch.nn.functional as F

from ..common import GAN, RNN, AnomalyDetector
from ..layers import RBFKernel
from ...models import BaseModel
from ...optim.loss import Loss
from ...optim.trainer import Trainer


class MADGAN(GAN, BaseModel):
    def __init__(self, input_dim: int, latent_dim: int = 15, generator_hidden_dims: List[int] = [100, 100, 100],
                 discriminator_hidden_dims: List[int] = [100]):
        generator = torch.nn.Sequential(
            RNN('lstm', 's2s', latent_dim, generator_hidden_dims),
            torch.nn.Linear(generator_hidden_dims[-1], input_dim),
            torch.nn.Sigmoid()  # For data to be in range [0, 1]
        )
        discriminator = torch.nn.Sequential(
            RNN('lstm', 's2s', input_dim, discriminator_hidden_dims),
            torch.nn.Linear(discriminator_hidden_dims[-1], 1)
        )

        super(MADGAN, self).__init__(generator, discriminator)

        self.latent_dim = latent_dim

    def grouped_parameters(self) -> Tuple[Iterator[torch.nn.Parameter], ...]:
        return self.discriminator.parameters(), self.generator.parameters()


class MADGANTrainer(Trainer):
    def __init__(self, *args, disc_iterations: int = 1, gen_iterations: int = 3, **kwargs):
        super(MADGANTrainer, self).__init__(*args, **kwargs)

        self.disc_iterations = disc_iterations
        self.gen_iterations = gen_iterations

    def validate_batch(self, network: MADGAN, val_metrics: Dict[str, Callable],
                       b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> Dict[str, float]:
        real_x, = b_inputs
        real_target, = b_targets

        # Generate a random vector for z
        real_z = torch.randn(*real_x.shape[:2], network.latent_dim, dtype=real_x.dtype, device=real_x.device)
        # (T, B, latent)

        b_inputs = (real_z, real_x)

        return super(MADGANTrainer, self).validate_batch(network, val_metrics, b_inputs, b_targets, *args, **kwargs)

    def train_batch(self, network: MADGAN, losses: List[Loss], optimizers: List[torch.optim.Optimizer], epoch: int,
                    num_epochs: int, b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...]) \
            -> List[float]:
        real_x, = b_inputs
        real_target, = b_targets

        disc_loss, gen_loss = losses
        disc_opt, gen_opt = optimizers

        # Generate a random vector for z
        real_z = torch.randn(*real_x.shape[:2], network.latent_dim, dtype=real_x.dtype, device=real_x.device)
        # (T, B, latent)

        for _ in range(self.disc_iterations):
            # Train Discriminator for x
            disc_loss_val, = super(MADGANTrainer, self).train_batch(network, [disc_loss], [disc_opt], epoch, num_epochs,
                                                                (real_z, real_x), (real_target,))

        for _ in range(self.gen_iterations):
            # Generate a random vector for z
            real_z = torch.randn(*real_x.shape[:2], network.latent_dim, dtype=real_x.dtype, device=real_x.device)
            # (T, B, latent)

            gen_loss_val, = super(MADGANTrainer, self).train_batch(network, [gen_loss], [gen_opt], epoch, num_epochs,
                                                                   (real_z, real_x), b_targets)

        return [disc_loss_val, gen_loss_val]


class MADGANAnomalyDetector(AnomalyDetector):
    def __init__(self, model: MADGAN, max_iter: int = 1000, lambder: float = 0.5, rec_error_tolerance: float = 0.1):
        super(MADGANAnomalyDetector, self).__init__()

        self.model = model
        self.max_iter = max_iter
        self.lambder = lambder
        self.rec_error_tolerance = rec_error_tolerance
        self.rbf_kernel = RBFKernel()

    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor) -> torch.Tensor:
        # Input shape (T, B, D)
        # Output shape (B, B)
        x = x.transpose(0, 1)
        x = x.reshape(x.shape[0], -1)
        result = torch.matmul(x, x.T)
        diag = torch.diag(result).unsqueeze(-1)
        result *= -2

        return diag + result + diag.T

    def _kernel_dissimilarity_error(self, x_gen: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        similarity = self.rbf_kernel(x_gen, x_true, diag_only=True)
        return torch.mean(1 - similarity)

    def _optimize_reconstruction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        # Generate random latent variable and set requires_grad=True so that we can optimize over it
        z = torch.randn(*x.shape[:-1], self.model.latent_dim, dtype=x.dtype, device=x.device, requires_grad=True)
        optimizer = torch.optim.Adam([z])
        # We need to put the network into training mode, otherwise CuDNN will throw an error. CPU is fine
        # even if we leave the net in eval mode
        self.model.generator.train()

        # Compute a heuristic bandwidth for the RBF kernel that is used in the reconstruction error measure
        sigma = torch.median(self._pairwise_sq_dist(x))
        gamma = 1 / (2 * sigma ** 2)
        self.rbf_kernel.gamma = gamma.item()

        for i in range(self.max_iter):
            optimizer.zero_grad(set_to_none=True)
            x_gen = self.model.generator(z)
            rec_error = self._kernel_dissimilarity_error(x_gen, x)
            if rec_error < self.rec_error_tolerance:
                break

            rec_error.backward(inputs=[z])
            optimizer.step()

            # Clamp z
            torch.clip(z, -100, 100)

        x_gen = torch.nan_to_num(x_gen.clone().detach())

        # Disable Training mode again
        self.model.generator.eval()

        return z.clone().detach(), x_gen

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input shape (T, B, D)
        # Output shape (B,)
        x, = inputs
        with torch.no_grad():
            disc_score = self.model.discriminator(x)
        # Use only score for the last point
        disc_score = F.binary_cross_entropy_with_logits(disc_score[-1], torch.ones_like(disc_score[-1]))

        z, x_gen = self._optimize_reconstruction(x)
        abs_error = torch.mean(torch.abs(x[-1] - x_gen[-1]), dim=-1)

        return self.lambder * abs_error + (1 - self.lambder) * disc_score

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[-1]
