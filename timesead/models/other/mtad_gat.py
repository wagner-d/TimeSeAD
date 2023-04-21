from typing import Union, Tuple, Sequence

import torch
import torch.nn.functional as F
import tqdm

from ..common import MLP, VAE, VAELoss, AnomalyDetector, DenseVAEEncoder
from ...models import BaseModel


class GAT(torch.nn.Module):
    def __init__(self, num_nodes: int, node_size: int, initializer_range: float = 0.02):
        super(GAT, self).__init__()

        weight = torch.randn(2*node_size)
        weight *= initializer_range
        self.weight = torch.nn.Parameter(weight)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.layer_norm = torch.nn.LayerNorm([num_nodes, num_nodes], eps=1e-14)
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_nodes, n_features = x.shape
        # (B, n_nodes, n_features) -> (B, n_nodes, n_nodes, 2*n_features)
        attention = torch.cat([torch.stack([x]*n_nodes, dim=1), torch.stack([x]*n_nodes, dim=2)], dim=-1)
        # (B, n_nodes, n_nodes, 2*n_features) * (2*n_features) -> (B, n_nodes, n_nodes)
        attention = torch.inner(attention, self.weight)
        attention = self.layer_norm(attention)
        attention = self.leaky_relu(attention)
        attention = self.softmax(attention)

        # (B, n_nodes, n_nodes) * (B, n_nodes, n_features) -> (B, n_nodes, n_features)
        out = torch.matmul(attention, x)
        return torch.sigmoid(out)


class MTAD_GAT(BaseModel):
    def __init__(self, input_features: int, window_size: int = 100, gru_hidden_dim: int = 300,
                 gru_dropout_prob: float = 0.0, mlp_hidden_dim: Union[int, Sequence[int]] = (300, 300, 300),
                 vae_hidden_dim: int = 300):
        super(MTAD_GAT, self).__init__()

        self.conv_layer = torch.nn.Conv1d(input_features, input_features, kernel_size=(7,), padding=3)
        self.feature_gat = GAT(input_features, window_size)
        self.temporal_gat = GAT(window_size, input_features)
        self.gru_layer = torch.nn.GRU(3*input_features, hidden_size=gru_hidden_dim, batch_first=True,
                                      dropout=gru_dropout_prob)
        self.forecast_MLP = MLP(gru_hidden_dim, mlp_hidden_dim, input_features, activation=torch.nn.ReLU())
        self.vae_encoder = DenseVAEEncoder(gru_hidden_dim, [vae_hidden_dim], vae_hidden_dim)
        self.vae_decoder = DenseVAEEncoder(vae_hidden_dim, [vae_hidden_dim], input_features)
        self.vae = VAE(self.vae_encoder, self.vae_decoder, logvar_out=False)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x, = inputs
        B, T, D = x.shape
        # (B, T, D) -> (B, D, T)
        conv_out = self.conv_layer(torch.transpose(x, 1, 2))

        # Note that the input is (B, D, T), so we have to transpose the result for the feature-oriented GAT
        feat_out = self.feature_gat(conv_out)
        feat_out = torch.transpose(feat_out, 1, 2)

        conv_out = conv_out.transpose(1, 2)
        temp_out = self.temporal_gat(conv_out)
        # both have shape (B, T, D)

        # -> (B, T, 3*D)
        concat_out = torch.cat([feat_out, temp_out, conv_out], dim=-1)

        # -> (B, T, gru_hidden_dim)
        gru_out, _ = self.gru_layer(concat_out)

        # MLP for forecasting
        # (B, gru_hidden_dim) -> (B, 1, D)
        mlp_prediction = self.forecast_MLP(gru_out[:, -1]).view(B, 1, D)

        # (B, gru_hidden_dim) -> (B, T * D)
        # It is not really specified in the paper if we only want the output of the last timestep here or the entire
        # sequence
        vae_z_mean, vae_z_std, vae_x_mean, vae_x_std = self.vae(gru_out)

        # (B, T * D) -> (B, T, D)
        # vae_x_mean = vae_x_mean.view(B, T, D)
        # vae_x_std = vae_x_std.view(B, T, D)

        return mlp_prediction, vae_z_mean, vae_z_std, vae_x_mean, vae_x_std


class MTAD_GATLoss(VAELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MTAD_GATLoss, self).__init__(size_average, reduce, reduction, logvar_out=False)
        self.mse_loss = torch.nn.MSELoss(size_average, reduce, reduction)

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        pred_x, z_mean, z_std, x_dec_mean, x_dec_std = predictions
        next_x, curr_x = targets

        vae_loss = super(MTAD_GATLoss, self).forward((z_mean, z_std, x_dec_mean, x_dec_std),
                                                     (curr_x,))
        pred_loss = self.mse_loss(pred_x, next_x)

        # For some reason the paper does not have a trade-off coefficient here
        return vae_loss + pred_loss


class MTAD_GATAnomalyDetector(AnomalyDetector):
    def __init__(self, model: MTAD_GAT, gamma: float = 0.8):
        super(MTAD_GATAnomalyDetector, self).__init__()

        self.model = model
        self.gamma = gamma

    @staticmethod
    def compute_vae_online_anomaly_score(inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # We compute not only the log likelihood of the data, but the entire ELBO
        # x: (B, T, D)
        # output (B,)
        x, z_mu, z_std, x_dec_mean, x_dec_std = inputs

        # Compute MC approximation of Log likelihood
        nll_output = F.gaussian_nll_loss(x_dec_mean[:, -1, :], x[:, -1, :], x_dec_std[:, -1, :]**2,
                                         reduction='none', full=True)
        # We want the actual density instead of neg log density
        torch.mul(nll_output, -1, out=nll_output)
        # Clip to avoid numerical issues for float 32 everything beyond those borders is 0 or inf anyway
        # torch.clip(nll_output, min=-104, max=88)
        # expm1(x) = exp(x) - 1
        torch.expm1(nll_output, out=nll_output)
        # Compute 1 - p, doesn't really make much sense, though, since p is a density and not a probability
        torch.mul(nll_output, -1, out=nll_output)

        # For some reason densities are added in the exp space
        nll_output = torch.sum(nll_output, dim=-1)
        torch.nan_to_num(nll_output, out=nll_output)

        return nll_output

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D), x_next: (B, 1, D)
        x, x_next = inputs

        with torch.no_grad():
            x_pred, z_mu, z_std, x_dec_mean, x_dec_std = self.model((x,))

        vae_score = self.compute_vae_online_anomaly_score((x, z_mu, z_std, x_dec_mean, x_dec_std))

        pred_error = x_next - x_pred
        torch.square(pred_error, out=pred_error)
        pred_score = torch.sum(pred_error, dim=(-2, -1))

        # Both (B,)
        return vae_score, pred_score

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        # Collect all labels and scores from the dataset
        labels, rec_scores, pred_scores = [], [], []

        with torch.no_grad():
            for batch in tqdm.tqdm(dataset):
                batch_inputs, batch_labels = batch
                batch_inputs = tuple(b_in.to(self.dummy.device) for b_in in batch_inputs)
                batch_labels = tuple(b_l.to(self.dummy.device) for b_l in batch_labels)

                x, = batch_inputs
                label, x_next = batch_labels

                rec_score, pred_score = self.compute_online_anomaly_score((x, x_next))

                labels.append(label.squeeze(-1).cpu())
                rec_scores.append(rec_score.cpu())
                pred_scores.append(pred_score.cpu())

        labels = torch.cat(labels, dim=0)
        rec_scores = torch.cat(rec_scores, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)

        # Shift the scores so that they align correctly
        labels = labels[:-1]
        rec_scores = rec_scores[1:]
        pred_scores = pred_scores[:-1]

        # Combine the two scores
        torch.mul(pred_scores, self.gamma, out=pred_scores)
        torch.add(pred_scores, rec_scores, out=pred_scores)
        torch.div(pred_scores, 1 + self.gamma, out=pred_scores)

        assert labels.shape == pred_scores.shape

        return labels, pred_scores

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass