# Mostly taken from https://github.com/Francois-Aubet/gluon-ts/blob/adding_ncad_to_nursery/src/gluonts/nursery/ncad
#
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Original implementation taken and modified from
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
# distributed under the Apache Licence 2.0
# http://www.apache.org/licenses/LICENSE-2.0

from typing import Callable, Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..common import TCN, AnomalyDetector
from ...models import BaseModel
from ...data.transforms import Transform
from ...optim.loss import Loss
from ...optim.trainer import Trainer
from ...utils import torch_utils


class TCNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a Temporal Convolution Network (TCN).
    The computed representation is the output of a fully connected layer applied
    to the output of an adaptive max pooling layer applied on top of the TCN,
    which reduces the length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C_in`, `L`) where `B` is the
    batch size, `C_in` is the number of input channels, and `L` is the length of
    the input. Outputs a two-dimensional tensor (`B`, `C_out`), `C_in` is the
    number of input channels C_in=tcn_channels*

    Args:
        in_channels : Number of input channels.
        out_channels : Dimension of the output representation vector.
        kernel_size : Kernel size of the applied non-residual convolutions.
        tcn_channels : Number of channels manipulated in the causal CNN.
        tcn_layers : Depth of the causal CNN.
        tcn_out_channels : Number of channels produced by the TCN.
            The TCN outputs a tensor of shape (B, tcn_out_channels, T)
        maxpool_out_channels : Fixed length to which each channel of the TCN
            is reduced.
        normalize_embedding : Normalize size of the embeddings
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        tcn_channels: int,
        tcn_layers: int,
        tcn_out_channels: int,
        maxpool_out_channels: int = 1,
        normalize_embedding: bool = True,
    ):

        super(TCNEncoder, self).__init__()

        dilations = [2**i for i in range(tcn_layers+1)]
        filters = [tcn_channels] * tcn_layers + [tcn_out_channels]
        tcn = TCN(
            input_dim=in_channels,
            nb_filters=filters,
            kernel_size=kernel_size,
            nb_stacks=1,
            dilations=dilations,
            padding='causal',
            use_skip_connections=False,
            dropout_rate=0.0,
            return_sequences=True,
            activation=torch.nn.LeakyReLU(),
            use_batch_norm=False,
            use_layer_norm=False
        )

        maxpool_out_channels = int(maxpool_out_channels)
        maxpooltime = torch.nn.AdaptiveMaxPool1d(maxpool_out_channels)
        flatten = torch.nn.Flatten()  # Flatten two and third dimensions (tcn_out_channels and time)
        fc = torch.nn.Linear(tcn_out_channels * maxpool_out_channels, out_channels)
        self.network = torch.nn.Sequential(tcn, maxpooltime, flatten, fc)

        self.normalize_embedding = normalize_embedding

    def forward(self, x):
        u = self.network(x)
        if self.normalize_embedding:
            return F.normalize(u, p=2, dim=1)
        else:
            return u


class ContrastiveClassifier(torch.nn.Module):
    """Contrastive Classifier.
    Calculates the distance between two random vectors, and returns an exponential transformation of it,
    which can be interpreted as the logits for the two vectors being different.
    p : Probability of x1 and x2 being different
    p = 1 - exp( -dist(x1,x2) )
    """
    def __init__(self, distance: Union[Callable[[torch.Tensor, torch.Tensor], str], torch.Tensor]):
        """
        Args:
            distance : A Function which takes two (batches of) vectors and returns a (batch of)
                positive number.
        """
        super().__init__()

        distance_functions = dict(l2_distance=l2_distance)
        if not callable(distance):
            distance = distance_functions[distance]

        self.distance = distance
        self.eps = 1e-10

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Compute distance
        dists = self.distance(x1, x2)

        # Probability of the two embeddings being equal: exp(-dist)
        log_prob_equal = -dists

        # Computation of log_prob_different
        log_prob_different = torch_utils.log1mexp(torch.clamp(log_prob_equal, max=-self.eps))

        logits_different = log_prob_different - log_prob_equal

        return logits_different


def l2_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Inputs are (B, D)
    diff = x1 - x2
    return torch_utils.batched_dot(diff, diff)


class NCAD(BaseModel):
    """Neural Contrastive Detection in Time Series"""

    def __init__(
        self,
        # hparams for the input data
        ts_channels: int,
        suspect_window_length: int = 1,
        # hparams for encoder
        tcn_kernel_size: int = 7,
        tcn_layers: int = 8,
        tcn_out_channels: int = 20,
        tcn_maxpool_out_channels: int = 8,
        embedding_rep_dim: int = 120,
        normalize_embedding: bool = True,
        # hparams for classifier
        distance: Union[Callable, str] = l2_distance,
    ) -> None:

        super().__init__()

        self.suspect_window_length = suspect_window_length

        # Encoder Network
        self.encoder = TCNEncoder(
            in_channels=ts_channels,
            out_channels=embedding_rep_dim,
            kernel_size=tcn_kernel_size,
            tcn_channels=tcn_out_channels,
            tcn_layers=tcn_layers,
            tcn_out_channels=tcn_out_channels,
            maxpool_out_channels=tcn_maxpool_out_channels,
            normalize_embedding=normalize_embedding,
        )

        # Contrast Classifier
        self.classifier = ContrastiveClassifier(
            distance=distance,
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (B, T, D)
        x, = inputs
        x = x.transpose(1, 2)

        ts_whole_embedding = self.encoder(x)
        ts_context_embedding = self.encoder(x[..., :-self.suspect_window_length])

        logits_anomaly = self.classifier(ts_whole_embedding, ts_context_embedding)

        return logits_anomaly


def coe_batch(x: torch.Tensor, y: torch.Tensor, coe_rate: float, suspect_window_length: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Contextual Outlier Exposure.

    Args:
        x : Tensor of shape (batch, time, D)
        y : Tensor of shape (batch, )
        coe_rate : Number of generated anomalies as proportion of the batch size.
    """

    if coe_rate == 0:
        raise ValueError(f"coe_rate must be > 0.")
    batch_size, window_size, ts_channels = x.shape
    oe_size = int(batch_size * coe_rate)
    device = x.device

    # Select indices
    idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,), device=device)
    # sample and apply nonzero offset to avoid fixed points
    idx_2 = torch.randint(low=1, high=batch_size, size=(oe_size,), device=device)
    idx_2 += idx_1
    torch.remainder(idx_2, batch_size, out=idx_2)

    if ts_channels > 3:
        numb_dim_to_swap = torch.randint(low=3, high=ts_channels, size=(oe_size,), device=device)
    else:
        numb_dim_to_swap = torch.ones(oe_size, dtype=torch.long, device=device) * ts_channels

    x_oe = x[idx_1].clone()  # .detach()
    oe_time_start_end = torch.randint(low=x.shape[-1] - suspect_window_length, high=x.shape[-1] + 1, size=(oe_size, 2),
                                      device=device)
    oe_time_start_end.sort(dim=1)
    # for start, end in oe_time_start_end:
    for i in range(oe_size):
        # obtain the dimensions to swap
        numb_dim_to_swap_here = numb_dim_to_swap[i].item()
        dims_to_swap_here = torch.from_numpy(np.random.choice(ts_channels, size=numb_dim_to_swap_here, replace=False))\
            .to(torch.long).to(device)

        # obtain start and end of swap
        start, end = oe_time_start_end[i]
        start, end = start.item(), end.item()

        # swap
        x_oe[i, start:end, dims_to_swap_here] = x[idx_2[i], start:end, dims_to_swap_here]

    # Label as positive anomalies
    y_oe = torch.ones(oe_size, dtype=y.dtype, device=device)

    return x_oe, y_oe


def mixup_batch(x: torch.Tensor, y: torch.Tensor, mixup_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x : Tensor of shape (batch, time, D)
        y : Tensor of shape (batch, )
        mixup_rate : Number of generated anomalies as proportion of the batch size.
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #
    device = x.device

    # Select indices
    idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,), device=device)
    # sample and apply nonzero offset to avoid fixed points
    idx_2 = torch.randint(low=1, high=batch_size, size=(mixup_size,), device=device)
    idx_2 += idx_1
    torch.remainder(idx_2, batch_size, out=idx_2)

    # sample mixing weights:
    beta_param = 0.05
    weights = torch.from_numpy(np.random.beta(beta_param, beta_param, (mixup_size,))).type_as(x).to(device)
    oppose_weights = 1.0 - weights

    # Create contamination
    x_mix_1 = x[idx_1]
    x_mix_2 = x[idx_2]  # Pretty sure that the index here should be idx_2 instead of idx_1
    x_mixup = x_mix_1 * weights[:, None, None]
    x_mixup.addcmul_(x_mix_2, oppose_weights[:, None, None])

    # Label as positive anomalies
    y_mixup = y[idx_1] * weights
    y_mixup.addcmul_(y[idx_2], oppose_weights)

    return x_mixup, y_mixup


class NCADTrainer(Trainer):
    def __init__(self, *args, coe_rate: float = 1.116, mixup_rate: float = 1.96, **kwargs):
        super(NCADTrainer, self).__init__(*args, **kwargs)

        self.coe_rate = coe_rate
        self.mixup_rate = mixup_rate
    
    def validate_batch(self, network: torch.nn.Module, val_metrics: Dict[str, Callable],
                       b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
        -> Dict[str, float]:
        x, = b_inputs
        y, = b_targets

        # Reduce labels. Label is 1 if any point in the suspect window is anomalous
        y, _ = torch.max(y[:, -network.suspect_window_length:], dim=1)
        # BCELoss requires float targets
        y = y.float()
        
        return super(NCADTrainer, self).validate_batch(network, val_metrics, (x,), (y,), *args, **kwargs)

    def train_batch(self, network: NCAD, losses: List[Loss], optimizers: List[torch.optim.Optimizer],
                    epoch: int, num_epochs: int, b_inputs: Tuple[torch.Tensor, ...],
                    b_targets: Tuple[torch.Tensor, ...]) -> List[float]:
        x, = b_inputs
        y, = b_targets

        # Reduce labels. Label is 1 if any point in the suspect window is anomalous
        y, _ = torch.max(y[:, -network.suspect_window_length:], dim=1)
        # BCELoss requires float targets
        y = y.float()

        if self.coe_rate > 0:
            x_oe, y_oe = coe_batch(
                x=x,
                y=y,
                coe_rate=self.coe_rate,
                suspect_window_length=network.suspect_window_length
            )
            # Add COE to training batch
            x = torch.cat((x, x_oe), dim=0)
            y = torch.cat((y, y_oe), dim=0)

        if self.mixup_rate > 0.0:
            x_mixup, y_mixup = mixup_batch(
                x=x,
                y=y,
                mixup_rate=self.mixup_rate,
            )
            # Add Mixup to training batch
            x = torch.cat((x, x_mixup), dim=0)
            y = torch.cat((y, y_mixup), dim=0)

        return super(NCADTrainer, self).train_batch(network, losses, optimizers, epoch, num_epochs, (x,), (y,))


class NCADAnomalyDetector(AnomalyDetector):
    def __init__(self, model: NCAD):
        super(NCADAnomalyDetector, self).__init__()

        self.model = model

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (B, T, D)
        x, = inputs

        with torch.no_grad():
            logit = self.model(inputs)

        # prob: (B,)
        prob = torch.sigmoid(logit)

        return prob

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        label, = targets

        return label[:, -1]
    

class LocalOutlierInjectionTransform(Transform):
    """
    Inject spikes based on local noise
    """

    def __init__(
        self,
        parent: Transform,
        max_duration_spike: int = 2,
        spike_multiplier_range: Tuple[float, float] = (0.5, 2.0),
        spike_value_range: Tuple[float, float] = (-np.inf, np.inf),
        area_radius: int = 100,
        num_spikes: int = 10,
    ):
        super(LocalOutlierInjectionTransform, self).__init__(parent)
        self.max_duration_spike = max_duration_spike
        self.spike_multiplier_range = spike_multiplier_range
        self.spike_value_range = spike_value_range
        self.area_radius = area_radius
        self.num_spikes = num_spikes

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)
        values = inputs[0]
        labels = targets[0]

        # Length and dimension of the TimeSeries
        T, ts_channels = values.shape

        num_spikes = self.num_spikes
        if self.num_spikes < 1.0:
            num_spikes = int(self.num_spikes * T) + 1

        # sample a random magnitude for each injected spike
        # random uniform in spike_multiplier_range
        spike_multiplier = torch.rand(num_spikes, ts_channels) \
                           * (self.spike_multiplier_range[1] - self.spike_multiplier_range[0]) \
                           + self.spike_multiplier_range[0]

        # random sign
        sign = 2 * torch.randint(0, 2, size=(num_spikes, ts_channels)) - 1
        # sign = 2 * (np.random.choice(self.direction_options, size=(num_spikes, ts_channels)) == "increase") - 1
        spike_multiplier *= sign

        duration_spike = torch.randint(1, self.max_duration_spike + 1, size=(num_spikes,))

        spike_location = torch.empty_like(duration_spike)
        for i in range(len(spike_location)):
            spike_location[i] = torch.randint(0, len(values) - duration_spike[i], size=(1,))

        label_spike = 1

        local_range = torch.zeros((num_spikes, 2, ts_channels))
        quantiles = torch.tensor([0.05, 0.95])
        for i, t in enumerate(spike_location):
            local_left = max(0, t - self.area_radius)
            local_right = min(t + self.area_radius, len(values) - 1)

            area = values[local_left:local_right]
            local_range[i] = torch.nanquantile(area, q=quantiles, dim=0)
        spike_addition = (local_range[:, 1, :] - local_range[:, 0, :]) * spike_multiplier
        assert spike_addition.shape == (num_spikes, ts_channels)

        ## Set some of the spikes to zero, so that there is not always an anomaly in each dimension
        if ts_channels > 3:
            F.dropout(spike_addition, p=0.35, training=True, inplace=True)
            # indices = np.random.choice(
            #     np.arange(spike_addition.size), replace=False, size=int(spike_addition.size * 0.35)
            # )
            #
            # spike_addition[indices // ts_channels - 1, indices // num_spikes - 1] = 0

        ## Prepare the spike to be added
        add_spike = torch.zeros_like(values)
        labels_addition = torch.zeros_like(labels)
        for i, t in enumerate(spike_location):
            add_spike[t : t + duration_spike[i]] = spike_addition[i]
            labels_addition[t : t + duration_spike[i]] = label_spike

        values_out = values + torch.clip(add_spike, *self.spike_value_range)
        labels_out = torch.logical_or(labels, labels_addition).to(torch.long)

        return (values_out,), (labels_out,)
