from typing import Tuple

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from timesead.models.reconstruction.lstm_ae import LSTMAE
from timesead.models.common import AnomalyDetector
from timesead.utils import torch_utils


class LSTMAEOCSVMAnomalyDetector(AnomalyDetector):
    def __init__(self, model: LSTMAE, kernel: str = 'rbf', gamma: float = 0.001, nu: float = 0.4,
                 normalize_data: bool = False):
        super(LSTMAEOCSVMAnomalyDetector, self).__init__()

        self.model = model
        self.ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        if normalize_data:
            self.ocsvm = make_pipeline(StandardScaler(), self.ocsvm)

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        latent_vectors = []
        self.model.eval()
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(torch_utils.get_device(self.model)) for b_inp in b_inputs)
            x, = b_inputs

            with torch.no_grad():
                z = self.model.encode(x)
                latent_vectors.append(z[-1].cpu().numpy())

        latent_vectors = np.concatenate(latent_vectors, axis=0)
        self.ocsvm.fit(latent_vectors)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, = inputs
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x)

        z = z[-1].cpu().numpy()
        # OCSVM labels normal points as 1 and anomalies as -1, so we have to invert the score
        return -torch.from_numpy(self.ocsvm.score_samples(z))

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (T, B), output of shape (B,)
        target, = targets
        # Just return the last label of the window
        return target[-1]
