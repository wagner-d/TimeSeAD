from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvLSTM, SameZeroPad2d
from ..common import MSEReconstructionAnomalyDetector, AnomalyDetector
from ...models import BaseModel
from ...data.transforms import Transform, WindowTransform
from ...optim.loss import Loss
from ...utils import torch_utils
from ...utils.utils import ceil_div


class MSCRED(BaseModel):
    """
    input is signature matrices of shape (Seq_len, Batch, Channel, Height, Width)
    """

    def __init__(self, n_features: int, in_channels: int, c_out: int = 256, small_model: bool = False,
                 chi: float = 5.0):
        super(MSCRED, self).__init__()

        self.small_model = small_model
        self.chi = chi

        # Convolutional Encoder
        same_pad = SameZeroPad2d(2)
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        if small_model:
            self.enc2 = nn.Conv2d(32, c_out, kernel_size=2, stride=2, padding=0)
        else:
            self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.enc3 = nn.Sequential(same_pad, nn.Conv2d(64, 128, kernel_size=2, stride=2))
            self.enc4 = nn.Sequential(same_pad, nn.Conv2d(128, c_out, kernel_size=2, stride=2))

        # ConvLSTM
        spatial_dim = n_features
        self.lstm1 = ConvLSTM(32, 32, 3, (spatial_dim, spatial_dim))
        if small_model:
            spatial_dim = spatial_dim // 2
            self.lstm2 = ConvLSTM(c_out, c_out, 2, (spatial_dim, spatial_dim))
        else:
            spatial_dim = ceil_div(spatial_dim, 2)
            self.lstm2 = ConvLSTM(64, 64, 3, (spatial_dim, spatial_dim))
            spatial_dim = ceil_div(spatial_dim, 2)
            self.lstm3 = ConvLSTM(128, 128, 2, (spatial_dim, spatial_dim))
            spatial_dim = ceil_div(spatial_dim, 2)
            self.lstm4 = ConvLSTM(c_out, c_out, 2, (spatial_dim, spatial_dim))

        # Convolutional Decoder
        if small_model:
            self.dec1 = nn.ConvTranspose2d(c_out, 32, kernel_size=2, stride=2)
            self.dec2 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        else:
            # self.dec1 = nn.Sequential(same_pad, nn.ConvTranspose2d(c_out, 128, kernel_size=2, stride=2))
            self.dec1 = nn.ConvTranspose2d(c_out, 128, kernel_size=2, stride=2)
            # self.dec2 = nn.Sequential(same_pad, nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2))
            self.dec2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
            self.dec3 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1)
            self.dec4 = nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=1, padding=1)

    def _attention(self, tensor: torch.Tensor):
        """
        input of shape (seq_len, batch, channel, h, w)
        """
        vec = tensor.view(*tensor.shape[:2], -1)
        alpha = torch_utils.batched_dot(vec, vec[-1].unsqueeze(0)) / self.chi
        # Compute softmax along the sequence dimension to normalize attention weights
        alpha = F.softmax(alpha, dim=0)

        # Out shape [batch, channel, h, w]
        return torch.sum(tensor * alpha.view(*alpha.shape, 1, 1, 1), dim=0)

    def forward(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        # In Signature Matrix of shape: [h, BZ, num_wins, num_features, num_features]
        # If temp_corr is True, Out shape: [h, BZ, num_wins, max(win), max(win)]
        x, = inputs

        # Save batch size
        h, bz, num_wins, num_features, _ = x.shape
        # To avoid for loop over h treat as bigger batch
        # Shape [h*BZ, num_wins, num_features, num_features]
        x = x.view(-1, *x.shape[2:])

        out = {}
        # Convolutional Encoder
        # 1st layer
        out[0] = F.selu(self.enc1(x), inplace=True)
        # 2nd layer
        out[1] = F.selu(self.enc2(out[0]), inplace=True)
        if not self.small_model:
            # 3rd layer
            out[2] = F.selu(self.enc3(out[1]), inplace=True)
            # 4th layer
            out[3] = F.selu(self.enc4(out[2]), inplace=True)

        # Reshape outputs again to split h and batch size for the following LSTM layers
        # [h*BZ, channels, height, width] -> [h, BZ, channels, height, width]
        out[0] = out[0].view(h, bz, *out[0].shape[1:])
        out[1] = out[1].view(h, bz, *out[1].shape[1:])
        if not self.small_model:
            out[2] = out[2].view(h, bz, *out[2].shape[1:])
            out[3] = out[3].view(h, bz, *out[3].shape[1:])

        # ConvLSTM
        out[0] = self._attention(self.lstm1(out[0])[0])
        out[1] = self._attention(self.lstm2(out[1])[0])
        if not self.small_model:
            out[2] = self._attention(self.lstm3(out[2])[0])
            out[3] = self._attention(self.lstm4(out[3])[0])

        # Convolutional Decoder
        if self.small_model:
            out_x = F.selu(self.dec1(out[1], output_size=out[0].size()), inplace=True)
            out_x = torch.cat([out_x, out[0]], dim=1)
            return F.selu(self.dec2(out_x), inplace=True)
        else:
            target_size = (out[2].shape[-2] + 1, out[2].shape[-1] + 1)
            out_x = self.dec1(out[3], output_size=target_size)
            # Need to crop the padding that was added in the encoder
            out_x = F.selu(out_x[..., :-1, :-1], inplace=True)
            out_x = torch.cat([out_x, out[2]], dim=1)

            target_size = (out[1].shape[-2] + 1, out[1].shape[-1] + 1)
            out_x = self.dec2(out_x, output_size=target_size)
            # Need to crop the padding that was added in the encoder
            out_x = F.selu(out_x[..., :-1, :-1], inplace=True)
            out_x = torch.cat([out_x, out[1]], dim=1)

            out_x = F.selu(self.dec3(out_x, output_size=out[0].size()), inplace=True)
            out_x = torch.cat([out_x, out[0]], dim=1)

            return F.selu(self.dec4(out_x), inplace=True)


class MSCREDLoss(Loss):
    def __init__(self):
        super(MSCREDLoss, self).__init__()

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        x_pred, = predictions
        x_true, = targets

        # Input for MSCRED is as stack of signature matrices, i.e., tensor of shape (h, B, D, D). However,
        # we only predict the last signature matrix of shape (B, D, D).
        return self.mse_loss(x_pred, x_true[-1])


class MSCREDAnomalyDetector(MSEReconstructionAnomalyDetector):
    def __init__(self, model: MSCRED):
        """
        This is what Florian uses, but in the paper they describe sth. completely different.
        They compute the number of badly reconstructed entries in the signature matrix (i.e., higher than
        some threshold) and use that count as the anomaly score
        """

        super(MSCREDAnomalyDetector, self).__init__(model, batch_first=False)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (h, B, wins, D, D), output of shape (B,)
        with torch.no_grad():
            prediction = self.model(inputs)

        input, = inputs
        # Input for MSCRED is as stack of signature matrices, i.e., tensor of shape (h, B, wins,  D, D). However,
        # we only predict the last signature matrix of shape (B, wins, D, D).
        sq_error = torch.mean((input[-1] - prediction) ** 2, dim=(-3, -2, -1))

        return sq_error

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class MSCREDAnomalyDetectorOrig(AnomalyDetector):
    def __init__(self, model: MSCRED, error_threshold: float = 0.5):
        super(MSCREDAnomalyDetectorOrig, self).__init__()

        self.model = model
        self.error_threshold = error_threshold

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (h, B, wins, D, D), output of shape (B,)
        with torch.no_grad():
            prediction = self.model(inputs)

        input, = inputs
        # Input for MSCRED is as stack of signature matrices, i.e., tensor of shape (h, B, wins,  D, D). However,
        # we only predict the last signature matrix of shape (B, wins, D, D).
        error = input[-1] - prediction
        error.abs_()
        # We compute the number of entries in the signature matrix that have a higher reconstruction error than some
        # threshold
        num_badly_reconstructed = torch.sum(error > self.error_threshold, dim=(-3, -2, -1))

        return num_badly_reconstructed

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B) output of shape (B)
        target, = targets
        # Just return the last label of the window
        return target[-1]


def compute_signature_matrix(x: torch.Tensor, seg_interval: int, wins: Tuple[int, ...], h: int) -> torch.Tensor:
    # In shape: [T, D]
    # Get data
    X = [
        x[(s * seg_interval):(s * seg_interval + max(wins))]
        for s in range(h)
    ]
    X = torch.stack(X, dim=0)
    # Flatten features
    X = X.view(*X.shape[:2], -1)

    sig_mat = []

    for w in wins:
        sig_mat.append(torch.matmul(X[:, -w:, :].transpose(-1, -2), X[:, -w:, :]) / w)

    # Out shape: [h, num_wins, num_features, num_features]
    # If temp_corr is True, Out shape: [h, num_wins, max(win), max(win)]
    return torch.stack(sig_mat, dim=1)


class SignatureMatrixTransform(WindowTransform):
    def __init__(self, parent: Transform, wins: Tuple[int] = (10, 30, 60), seg_interval: int = 10, h: int = 5):
        super(SignatureMatrixTransform, self).__init__(parent, max(wins) + (h-1)*seg_interval, step_size=1)

        self.wins = wins
        self.seg_interval = seg_interval
        self.h = h

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = super(SignatureMatrixTransform, self)._get_datapoint_impl(item)

        inputs = tuple(compute_signature_matrix(x, self.seg_interval, self.wins, self.h) for x in inputs)

        return inputs, targets

    @property
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return len(self.wins), self.parent.num_features, self.parent.num_features
