from typing import List, Tuple, Type, Union, Iterator

import torch
from torch.nn import Parameter

from ..common import AE, GAN, GANDiscriminatorLoss, MSEReconstructionAnomalyDetector
from ..layers import ConvBlock
from ...models import BaseModel
from ...data.transforms import Transform
from ...optim.loss import Loss

class ConvEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, filters: List[int], conv_parameters: List[Tuple[int, int, int, bool, bool]],
                 block: Type[ConvBlock] = ConvBlock, conv_layer=torch.nn.Conv1d, activation=torch.nn.Identity()):
        super(ConvEncoder, self).__init__()

        dims = [input_dim] + filters
        modules = []
        for in_channels, out_channels, (kernel_size, stride, padding, bias, batch_norm) \
                in zip(dims[:-1], dims[1:], conv_parameters):
            layer = conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            modules.append(block(layer, out_channels, activation, batch_norm))

        self.layers = torch.nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int]]]:
        # x: (B, D, T)
        pre_conv_sizes = []
        for block in self.layers:
            pre_conv_sizes.append(x.shape[2:])
            x = block(x)

        return x, pre_conv_sizes


class ConvDecoder(torch.nn.Module):
    def __init__(self, input_dim: int, filters: List[int], conv_parameters: List[Tuple[int, int, int, bool, bool]],
                 block: Type[ConvBlock] = ConvBlock, conv_layer=torch.nn.ConvTranspose1d, activation=torch.nn.Identity()):
        super(ConvDecoder, self).__init__()

        dims = [input_dim] + filters
        modules = []
        for in_channels, out_channels, (kernel_size, stride, padding, bias, batch_norm) \
                in zip(dims[:-2], dims[1:-1], conv_parameters):
            layer = conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            modules.append(block(layer, out_channels, activation, batch_norm))

        # Don't use activation for last layer
        layer = conv_layer(dims[-2], dims[-1], conv_parameters[-1][0], conv_parameters[-1][1], conv_parameters[-1][2],
                           bias=conv_parameters[-1][3])
        modules.append(block(layer, dims[-1], torch.nn.Identity(), batch_norm=conv_parameters[-1][4]))

        self.layers = torch.nn.ModuleList(modules)

    def forward(self, inputs: Tuple[torch.Tensor, List[Tuple[int]]]) -> torch.Tensor:
        # x: (B, D_l, T_l)
        x, pre_conv_sizes = inputs
        for conv_block, pre_conv_size in zip(self.layers, pre_conv_sizes[::-1]):
            x = conv_block(x, output_size=pre_conv_size)

        return x


class BeatGANConvEncoder(ConvEncoder):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50,
                 last_kernel_size: int = 10, return_features: bool = False):
        filters = [
            conv_filters,
            conv_filters * 2,
            conv_filters * 4,
            conv_filters * 8,
            conv_filters * 16
        ]
        conv_params = [
            # (kernel_size, stride, padding, bias, batch_norm)
            (4, 2, 1, False, False),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
        ]
        super(BeatGANConvEncoder, self).__init__(input_dim, filters, conv_params, conv_layer=torch.nn.Conv1d,
                                                 activation=torch.nn.LeakyReLU(0.2, True))

        self.return_features = return_features

        self.last_conv = torch.nn.Conv1d(conv_filters * 16, latent_dim, last_kernel_size, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Union[List[Tuple[int]], torch.Tensor]]:
        x, sizes = super(BeatGANConvEncoder, self).forward(x)

        sizes.append(x.shape[2:])
        x_last = self.last_conv(x)

        if self.return_features:
            return x_last, x

        return x_last, sizes


class BeatGANConvDecoder(ConvDecoder):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50, last_kernel_size: int = 10):
        filters = [
            conv_filters * 16,
            conv_filters * 8,
            conv_filters * 4,
            conv_filters * 2,
            conv_filters,
            input_dim
        ]
        conv_params = [
            # (kernel_size, stride, padding, bias, batch_norm)
            (last_kernel_size, 1, 0, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
        ]
        super(BeatGANConvDecoder, self).__init__(latent_dim, filters, conv_params, conv_layer=torch.nn.ConvTranspose1d,
                                                 activation=torch.nn.ReLU(True))

        # we replace tanh by sigmoid, because we do 0-1 normalization
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, inputs: Tuple[torch.Tensor, List[Tuple[int]]]) -> torch.Tensor:
        x = super(BeatGANConvDecoder, self).forward(inputs)
        return self.final_activation(x)


class BeatGANConvAE(AE):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50, last_kernel_size: int = 10):
        encoder = BeatGANConvEncoder(input_dim, conv_filters, latent_dim, last_kernel_size)
        decoder = BeatGANConvDecoder(input_dim, conv_filters, latent_dim, last_kernel_size)

        super(BeatGANConvAE, self).__init__(encoder, decoder, return_latent=False)


class BeatGANModel(BaseModel, GAN):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50, last_kernel_size: int = 10):
        # Note: BeatGAN will only work with a window size of exactly 320
        generator = BeatGANConvAE(input_dim, conv_filters, latent_dim, last_kernel_size)
        discriminator = BeatGANConvEncoder(input_dim, conv_filters, 1, last_kernel_size, return_features=True)
        super(BeatGANModel, self).__init__(generator, discriminator)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # x: (B, T, D)
        x, = inputs
        x = x.transpose(1, 2)
        fake_x, (real_x_score, real_x_features), (fake_x_score, fake_x_features) = super(BeatGANModel, self).forward((x, x))
        fake_x = fake_x.transpose(1, 2)

        return fake_x, real_x_score, real_x_features, fake_x_score, fake_x_features

    def grouped_parameters(self) -> Tuple[Iterator[Parameter], ...]:
        return self.discriminator.parameters(), self.generator.parameters()


class BeatGANDiscriminatorLoss(GANDiscriminatorLoss):
    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, real_x_features, fake_x_score, fake_x_features = predictions
        return super(BeatGANDiscriminatorLoss, self).forward((fake_x, real_x_score, fake_x_score), targets)


class BeatGANGeneratorLoss(Loss):
    def __init__(self, adversarial_weight: float = 1.0):
        super(BeatGANGeneratorLoss, self).__init__()

        self.adversarial_weight = adversarial_weight
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, real_x_features, fake_x_score, fake_x_features = predictions
        real_x, = targets

        rec_loss_inputs = self.mse_loss(fake_x, real_x)
        rec_loss_features = self.mse_loss(fake_x_features, real_x_features)

        return rec_loss_inputs + self.adversarial_weight * rec_loss_features


class BeatGANReconstructionAnomalyDetector(MSEReconstructionAnomalyDetector):
    def __init__(self, model: BeatGANModel):
        super(BeatGANReconstructionAnomalyDetector, self).__init__(model, batch_first=True)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (B, T, D)
        x, = inputs
        x = x.transpose(1, 2)
        with torch.no_grad():
            fake_x = self.model.generator(x)

        # Note that x is now (B, D, T)
        sq_error = torch.mean((x - fake_x) ** 2, dim=1)

        return sq_error[:, -1]

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class WrapAugmentTransform(Transform):
    def __init__(self, parent: Transform, distort_fraction: float = 0.05, n_augmentations: int = 1):
        """
        Implements BeatGANs time-series distortion. This should be applied after windowing.

        :param parent: This transform's parent.
        :param distort_fraction: Fraction of time points that should be distorted. Note that 2 distortions are applied,
            so in the end distor_fraction*2 data points will be distorted
        :param n_data_augmentations: For each original time-series in parent, this will produce n_data_augmentations
                additional augmented time series
        """
        super(WrapAugmentTransform, self).__init__(parent)

        self.distort_fraction = distort_fraction
        self.n_augmentations = n_augmentations

    def aug_ts(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        # output (T, D)
        window_length = x.shape[0]
        left_window_length = window_length // 2
        right_window_length = window_length - left_window_length - 1

        distort_amount = int(0.5 * window_length * self.distort_fraction)

        left_indices = torch.multinomial(torch.full((left_window_length,), 1 / left_window_length),
                                         2 * distort_amount, replacement=False)
        right_indices = torch.multinomial(torch.full((right_window_length,), 1 / right_window_length),
                                          2 * distort_amount, replacement=False)
        right_indices += left_window_length

        left_up_ticks = left_indices[:distort_amount]
        left_down_ticks = left_indices[distort_amount:2 * distort_amount]
        right_up_ticks = right_indices[:distort_amount]
        right_down_ticks = right_indices[distort_amount:2 * distort_amount]

        x_1 = torch.zeros_like(x)
        j = 0
        for i in range(window_length):
            if i in left_down_ticks or i in right_down_ticks:
                continue
            elif i in left_up_ticks or i in right_up_ticks:
                x_1[j] = x[i]
                j += 1
                x_1[j] = (x[i] + x[i + 1]) / 2
                j += 1
            else:
                x_1[j] = x[i]
                j += 1
        return x_1

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        orig_item, return_orig = divmod(item, 1 + self.n_augmentations)
        inputs, targets = self.parent.get_datapoint(orig_item)
        # If the index matches an original datapoint, match that instead
        if return_orig == 0:
            return inputs, targets

        # Otherwise compute an augmented version
        inputs = tuple(self.aug_ts(inp) for inp in inputs)
        return inputs, targets

    def __len__(self) -> int:
        return len(self.parent) * (1 + self.n_augmentations)
