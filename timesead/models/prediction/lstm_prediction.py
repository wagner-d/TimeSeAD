from typing import List, Union, Callable, Tuple

import torch
from torch.nn import functional as F

from ..common import RNN, MLP, PredictionAnomalyDetector
from ...models import BaseModel
from ...data.transforms import PredictionTargetTransform, Transform
from ...utils import torch_utils
from ...utils.utils import halflife2alpha


class LSTMPrediction(BaseModel):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [30, 20], linear_hidden_layers: List[int] = [],
                 linear_activation: Union[Callable, str] = torch.nn.ELU(), prediction_horizon: int = 3):
        """
        LSTM prediction (Malhotra2015)
        :param input_dim:
        :param lstm_hidden_dims:
        :param linear_hidden_layers:
        :param linear_activation:
        :param prediction_horizon:
        """
        super(LSTMPrediction, self).__init__()

        self.prediction_horizon = prediction_horizon

        self.lstm = RNN('lstm', 's2fh', input_dim, lstm_hidden_dims)
        self.mlp = MLP(lstm_hidden_dims[-1], linear_hidden_layers, prediction_horizon * input_dim, linear_activation)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (T, B, D)
        x, = inputs

        # hidden: (B, hidden_dims)
        hidden = self.lstm(x)
        # x_pred: (B, horizon * D)
        x_pred = self.mlp(hidden)
        # x_pred: (B, horizon, D)
        x_pred = x_pred.view(x_pred.shape[0], self.prediction_horizon, -1)
        # output: (horizon, B, D)
        return x_pred.transpose(0, 1)


class LSTMS2SPrediction(BaseModel):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [30, 20], linear_hidden_layers: List[int] = [],
                 linear_activation: Union[Callable, str] = torch.nn.ELU(), dropout: float = 0.0):
        """
        LSTM prediction (Filonov2016)
        :param input_dim:
        :param lstm_hidden_dims:
        :param linear_hidden_layers:
        :param linear_activation:
        """
        super(LSTMS2SPrediction, self).__init__()

        self.lstm = RNN('lstm', 's2s', input_dim, lstm_hidden_dims, dropout=dropout)
        self.mlp = MLP(lstm_hidden_dims[-1], linear_hidden_layers, input_dim, linear_activation)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (T, B, D)
        x, = inputs

        # hidden: (T, B, hidden_dims)
        hidden = self.lstm(x)
        # x_pred: (T, B, D)
        x_pred = self.mlp(hidden)

        return x_pred


class LSTMPredictionAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: LSTMPrediction):
        """
        Malhotra2016

        :param model:
        """
        super(LSTMPredictionAnomalyDetector, self).__init__()

        self.model = model

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        errors = []

        # Compute mean and covariance over the entire validation dataset
        counter = 0
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)
            with torch.no_grad():
                pred = self.model(b_inputs)

            target, = b_targets

            error = target - pred
            for j in range(error.shape[0]):
                for j2 in range(error.shape[1]):
                    index = counter + j + j2
                    if len(errors) < index + 1:
                        errors.append([])
                    errors[counter + j + j2].append(error[j, j2])
            counter += error.shape[1]

        errors = errors[self.model.prediction_horizon - 1:-self.model.prediction_horizon + 1]

        errors = torch_utils.nested_list2tensor(errors)
        errors = errors.view(errors.shape[0], -1)
        mean = torch.mean(errors, dim=0)
        errors -= mean
        cov = torch.matmul(errors.T, errors)
        cov /= errors.shape[0] - 1

        # Add a small epsilon to the diagonal of the matrix to make it non-singular
        cov.diagonal().add_(1e-5)
        # This construction ensures that the resulting precision matrix is pos. semi-definite, even if the condition
        # number of the cov matrix is large

        try:
            cholesky = torch.linalg.cholesky(cov)
        except (torch.linalg.LinAlgError, RuntimeError):
            # If the covariance matrix is not positive definite, we can try to add a small value to the diagonal
            # until it becomes positive definite
            for _ in range(100):
                cov.diagonal().add_(1e-4)
                try:
                    cholesky = torch.linalg.cholesky(cov)
                    break
                except (torch.linalg.LinAlgError, RuntimeError):
                    continue
            else:
                raise RuntimeError('Could not compute a valid covariance matrix!')
        precision = cov
        torch.cholesky_inverse(cholesky, out=precision)

        self.register_buffer('mean', mean)
        self.register_buffer('precision', precision)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        errors = []
        labels = []
        # Compute mean and covariance over the entire validation dataset
        counter = 0
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)
            with torch.no_grad():
                pred = self.model(b_inputs)

            label, target = b_targets

            error = target - pred
            for j in range(error.shape[0]):
                for j2 in range(error.shape[1]):
                    index = counter + j + j2
                    if len(errors) <= index:
                        errors.append([])
                    errors[counter + j + j2].append(error[j, j2])
            counter += error.shape[1]

            labels.append(label[-1].cpu())

        errors = errors[self.model.prediction_horizon - 1:-self.model.prediction_horizon + 1]
        errors = torch_utils.nested_list2tensor(errors)
        errors = errors.view(errors.shape[0], -1)
        labels = torch.cat(labels, dim=0)
        labels = labels[:-self.model.prediction_horizon + 1]

        errors -= self.mean
        scores = F.bilinear(errors, errors, self.precision.unsqueeze(0)).squeeze(-1)

        assert labels.shape == scores.shape

        return labels, scores.cpu()


class LSTMS2SPredictionAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: LSTMS2SPrediction, half_life: int):
        """
        Filonov2016

        :param model:
        :param half_life:
        """
        super(LSTMS2SPredictionAnomalyDetector, self).__init__()

        self.model = model
        self.alpha = halflife2alpha(half_life)


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, torch.Tensor, float, float]) \
            -> Tuple[torch.Tensor, float, float]:
        # x: (T, B, D), target: (T, B, D), moving_avg: ()
        x, target, moving_avg_num, moving_avg_denom = inputs

        with torch.no_grad():
            x_pred = self.model((x,))

        sq_error = target - x_pred
        torch.square(sq_error, out=sq_error)
        sq_error = torch.sum(sq_error, dim=-1)

        T, B = sq_error.shape
        sq_error = sq_error.T.flatten()
        moving_avg_num, moving_avg_denom = torch_utils.exponential_moving_avg_(sq_error, self.alpha,
                                                                               avg_num=moving_avg_num,
                                                                               avg_denom=moving_avg_denom)

        return sq_error.view(B, T).T, moving_avg_num, moving_avg_denom

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        errors = []
        labels = []
        moving_avg_num = 0
        moving_avg_denom = 0

        # Compute exp moving average of error score
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            x, = b_inputs
            label, target = b_targets

            sq_error, moving_avg_num, moving_avg_denom = self.compute_online_anomaly_score((x, target, moving_avg_num,
                                                                                            moving_avg_denom))
            errors.append(sq_error)
            labels.append(label.cpu())

        scores = torch.cat(errors, dim=1).transpose(0, 1).flatten()
        labels = torch.cat(labels, dim=1).transpose(0, 1).flatten()

        assert labels.shape == scores.shape

        return labels, scores.cpu()


class LSTMS2STargetTransform(PredictionTargetTransform):
    def __init__(self, parent: Transform, window_size: int, replace_labels: bool = False,
                 reverse: bool = False):
        super(LSTMS2STargetTransform, self).__init__(parent, window_size, window_size, replace_labels=replace_labels,
                                                     step_size=window_size, reverse=reverse)
