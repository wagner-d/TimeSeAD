import random
from typing import Tuple, Union

import torch
import tqdm

from ..common import AnomalyDetector
from ...models import BaseModel
from ...data.transforms import Transform
from ...optim.loss import LogCoshLoss


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = None, dropout: float = 0.):
        super().__init__()

        if dim_head is None:
            dim_head = dim

        self.dim_head = dim_head
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = torch.nn.Softmax(dim = -1)
        self.dropout = torch.nn.Dropout(dropout)

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*, T, D)

        # (B*, T, D) -> (B*, T, H * 3 * D')
        qkv = self.to_qkv(x)
        # (B*, T, H * 3 * D') -> (B*, H, T, 3 * D')
        qkv = qkv.view(*qkv.shape[:-1], self.heads, 3 * self.dim_head).transpose(-3, -2)
        # (B*, H, T, 3 * D') -> 3 * (B*, H, T, D')
        q, k, v = qkv.chunk(3, dim=-1)

        # (B*, H, T, D') x (B*, H, D', T) -> (B*, H, T, T)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # (B*, H, T, T) x (B*, H, T, D') -> (B*, H, T, D')
        out = torch.matmul(attn, v)
        # (B*, H, T, D') -> (B*, T, H * D')
        out = out.transpose(-3, -2).reshape(*x.shape[:-1], self.heads * self.dim_head)
        # (B*, T, H * D') -> (B*, T, D)
        return self.to_out(out)


class GENAD(BaseModel):
    def __init__(self, input_dim: int, window_size: int, split_folds: int = 5,
                 attention_heads: int = 12, attention_layers: int = 4, dropout: float = 0.0):
        super(GENAD, self).__init__()

        assert window_size % split_folds == 0

        self.window_size = window_size
        self.input_dim = input_dim
        self.split_folds = split_folds

        self.corr_attention_layers = torch.nn.ModuleList([
            Attention(window_size // split_folds, heads=attention_heads, dropout=dropout) for _ in range(attention_layers)
        ])

        self.ts_attention_layers = torch.nn.ModuleList([
            Attention(window_size // split_folds, heads=attention_heads, dropout=dropout) for _ in range(attention_layers)
        ])

        self.combination_weight = torch.nn.Parameter(torch.tensor(0.0))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T; D), mask_ind: (B, masked_dims)
        x, mask_ind = inputs
        B, T, D = x.shape
        masked_dims = mask_ind.shape[-1]

        assert T == self.window_size

        # Split input time series
        x_split = torch.tensor_split(x, self.split_folds, dim=-2)
        x_end = x_split[-1]

        mask_ind_b = torch.arange(B, dtype=torch.long, device=x.device).unsqueeze(1).expand(B, masked_dims)

        # Apply attention layers
        # Multi-correlation attention
        # (B, T', D) -> (B, D, T')
        x_rec_corr = x_end.transpose(-2, -1).clone()
        for layer in self.corr_attention_layers:
            x_rec_corr = layer(x_rec_corr)
        # (B, D, T') -> (B, T', masked_dims)
        x_rec_corr = x_rec_corr[mask_ind_b, mask_ind].transpose(-2, -1)

        # Time-series attention
        # Note: We could make this more efficient with a special attention implementation since we only need results
        # for the last fold
        # (B, T, D) -> (B, masked_dims, split_folds,  T')
        x_rec_ts_orig = x[mask_ind_b, :, mask_ind].view(B, masked_dims, self.split_folds, -1).clone()
        x_rec_ts = x_rec_ts_orig
        for layer in self.ts_attention_layers:
            x_rec_ts = layer(x_rec_ts)
            x_rec_ts[:, :, :-1, :] = x_rec_ts_orig[:, :, :-1, :]
        # (B, masked_dims, split_folds,  T') -> (B, T', masked_dims)
        x_rec_ts = x_rec_ts[:, :, -1, :].transpose(-2, -1)

        # Fuse reconstructions
        combination_weight = self.sigmoid(self.combination_weight)
        final_reconstruction = combination_weight * x_rec_corr + (1 - combination_weight) * x_rec_ts

        return final_reconstruction, (mask_ind_b, mask_ind)


class MaskedLogCoshLoss(LogCoshLoss):
    def __init__(self, split_folds: int = 5, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MaskedLogCoshLoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)

        self.split_folds = split_folds

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> \
        Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Use only the last fold and masked dimensions for reconstruction error
        # x_rec: (B, T // split_folds, masked_dims)
        x_rec, (mask_ind_b, mask_ind) = predictions
        # x_true: (B, T, D)
        x_true, = targets

        # (B, T, D) -> (B, masked_dims, T)
        x_true = x_true[mask_ind_b, :, mask_ind]
        # (B, masked_dims, T) -> (B, T // split_folds, masked_dims)
        x_true = torch.tensor_split(x_true, self.split_folds, dim=-1)[-1].transpose(-2, -1)

        return super(MaskedLogCoshLoss, self).forward((x_rec,), (x_true,))


class RandomMaskTransform(Transform):
    def __init__(self, parent: Transform, masked_fraction: float = 0.2, split_folds: int = 5):
        """
        Caches the results from a previous transform in memory so that expensive calculations do not have to be
        recomputed.

        :param parent: Another transform which is used as the data source for this transform.
        """
        super(RandomMaskTransform, self).__init__(parent)

        self.masked_dims = int(parent.num_features * masked_fraction)
        self.split_folds = split_folds

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)

        new_inputs = []
        # Randomly mask some dimensions in the last part of the TS
        mask_ind = torch.tensor(random.choices(range(self.parent.num_features), k=self.masked_dims + 1),
                                dtype=torch.long, device=inputs[0].device)
        replacement_ind = mask_ind[-1]
        mask_ind = mask_ind[:-1]

        for x in inputs:
            # x: (T; D)
            # Split input time series
            x_split = torch.tensor_split(x, self.split_folds, dim=0)
            x_end = x_split[-1]

            x_end[:, mask_ind] = x_end[:, replacement_ind:replacement_ind+1]
            new_inputs.append(x)

        new_inputs.append(mask_ind)

        return tuple(new_inputs), targets


class GENADDetector(AnomalyDetector):
    def __init__(self, model: GENAD, threshold_frac: float = 1.05):
        super(GENADDetector, self).__init__()

        self.model = model
        self.threshold_frac = threshold_frac

        self.loss = MaskedLogCoshLoss(split_folds=model.split_folds, reduction='none')

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (B, T, D)
        # output (B,)
        x, = inputs
        B, T, D = x.shape

        final_score = torch.zeros(B, dtype=torch.long, device=x.device)

        mask = torch.zeros((B, 1), dtype=torch.long, device=x.device)
        for d in range(D):
            mask[:, 0] = d

            with torch.no_grad():
                res = self.model((x, mask))

            # loss: (B,)
            loss = self.loss(res, (x,)).squeeze(-1)[:, -1]

            final_score += (loss > self.max_train_errors[d] * self.threshold_frac).to(torch.long)

        return final_score

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        max_errors = torch.zeros(self.model.input_dim, dtype=torch.float32, device=self.dummy.device)

        for batch in tqdm.tqdm(dataset):
            batch_inputs, batch_labels = batch
            batch_inputs = tuple(b_in.to(self.dummy.device) for b_in in batch_inputs)

            x, _ = batch_inputs
            B, T, D = x.shape

            mask = torch.zeros((B, 1), dtype=torch.long, device=x.device)
            for d in range(D):
                mask[:, 0] = d

                with torch.no_grad():
                    res = self.model((x, mask))

                # loss: (B, T)
                loss = self.loss(res, (x,))
                max_loss = torch.max(loss)

                torch.maximum(max_errors[d], max_loss, out=max_errors[d])

        self.register_buffer('max_train_errors', max_errors)

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[:, -1]
