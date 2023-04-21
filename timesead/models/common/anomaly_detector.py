import abc
from typing import Tuple

import torch
import tqdm

from ...models import BaseModel

class AnomalyDetector(torch.nn.Module, abc.ABC):
    def __init__(self):
        super(AnomalyDetector, self).__init__()

        # We use this empty parameter to determine the device of the detector
        self.register_buffer('dummy', torch.tensor([]), persistent=False)

    @abc.abstractmethod
    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute the online anomaly score for a batch of inputs. The output tensor must have the same shape as the
        output of `format_targets` when called with the corresponding targets for this batch. This method expects
        a window (or a batch of windows) as its input and should return a score for the last point in the window.

        :param inputs: tuple of input tensors
        :return: Tensor of shape (B,) that contains the anomaly scores for this batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute the offline anomaly score for a batch of inputs. The output tensor must have the same shape as the
        output of `format_targets` when called with the corresponding targets for this batch. This method expects
        a window (or a batch of windows) as its input and should return a score for the last point in the window.

        :param inputs: tuple of input tensors
        :return: Tensor of shape (N,) that contains the anomaly scores for this batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        """
        Fit this anomaly detector on a dataset. Note that we assume only normal data here.

        :param dataset: A dataset
        """
        raise NotImplementedError

    @abc.abstractmethod
    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Format the labels for a batch of targets. The output tensor must have the same shape as the
        output of `compute_online_anomaly_score` when called with the corresponding inputs for this batch.

        :param targets: tuple of target tensors
        :return: Tensor of shape (B,) that contains the ground truth labels for this batch
        """
        raise NotImplementedError

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.compute_online_anomaly_score(inputs)

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        # Collect all labels and scores from the dataset
        labels, scores = [], []

        for batch in tqdm.tqdm(dataset):
            batch_inputs, batch_labels = batch
            batch_inputs = tuple(b_in.to(self.dummy.device) for b_in in batch_inputs)

            batch_scores = self.compute_online_anomaly_score(batch_inputs)
            batch_labels = self.format_online_targets(batch_labels)

            labels.append(batch_labels.cpu())
            scores.append(batch_scores.cpu())

        labels = torch.cat(labels, dim=0)
        scores = torch.cat(scores, dim=0)

        assert labels.shape == scores.shape

        return labels, scores


class MSEReconstructionAnomalyDetector(AnomalyDetector):
    def __init__(self, model: BaseModel, batch_first: bool = True):
        super(MSEReconstructionAnomalyDetector, self).__init__()

        self.model = model
        self.batch_first = batch_first

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T, D) or (T, B, D), output of shape (B,)
        with torch.no_grad():
            prediction = self.model(inputs)

        input, = inputs
        sq_error = torch.mean((input - prediction)**2, dim=-1)

        return sq_error[:, -1] if self.batch_first else sq_error[-1]

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T) or (T, B), output of shape (B)
        target, = targets
        # Just return the last label of the window
        return target[:, -1] if self.batch_first else target[-1]


class MAEReconstructionAnomalyDetector(MSEReconstructionAnomalyDetector):
    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T, D) or (T, B, D), output of shape (B,)
        with torch.no_grad():
            prediction = self.model(inputs)

        input, = inputs
        abs_error = torch.mean(torch.abs(input - prediction), dim=-1)

        return abs_error[:, -1] if self.batch_first else abs_error[-1]

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class PredictionAnomalyDetector(AnomalyDetector, abc.ABC):
    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = []
        labels = []
        for b_inputs, b_targets in tqdm.tqdm(dataset):
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            x, = b_inputs
            label, x_target = b_targets

            label = self.format_online_targets(b_targets)
            score = self.compute_online_anomaly_score((x, x_target))

            scores.append(score.cpu())
            labels.append(label.cpu())

        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)

        assert labels.shape == scores.shape

        return labels, scores
