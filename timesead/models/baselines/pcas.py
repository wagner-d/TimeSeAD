from typing import Optional, Tuple, Union

from sklearn.decomposition import PCA, KernelPCA, SparsePCA

import torch

from ..common import AnomalyDetector
from ...data.statistics import get_data_all


class PCAAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        n_components: Union[int, float],
        pca_method: str = "standard",
        first_diffs: bool = False,
        cum_method: str = "mean",
        input_shape: str = "btf",
        **kwargs,
    ) -> None:
        """
        PCA Anomaly Detector
            A simlpe anomaly detector that uses PCA as reconstruction method.

        Args:
            n_components[int, float]: number of principle components to keep. If float with 0<n_comp<=1.,the number
                of components will equal the explained variance, note, that this only works for `standard` pca.
                Defaults equals 0.95.
            pca_method[str: {'standard','kernel'}]: Which PCA method should be used. Default is standard.
            first_diffs[bool]: Flag, if instead of raw values first difference should be used. Default is False.
            cum_method[str: {'mean','max'}]: Accumulation method over feature dimension. Note that when
                `feature_index` is not None. Then accumulation method will be ignored. Default is 'mean'.
        """
        super(PCAAnomalyDetector, self).__init__()

        self.first_diffs = first_diffs
        if cum_method.lower() not in ["mean", "max"]:
            raise ValueError("`cum_method` must be one of {'mean','max'}")
        self.cum_method = cum_method.lower()
        self.input_shape = input_shape

        if pca_method.lower() not in ["standard", "kernel"]:
            raise ValueError(
                "`pca_method` must be one of {'standard','kernel'}"
            )

        if pca_method.lower() == "kernel":
            self.model = KernelPCA(n_components, fit_inverse_transform=True, **kwargs)
        else:
            self.model = PCA(n_components, **kwargs)

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:

        data_all = get_data_all(dataset.dataset, take_fd=self.first_diffs)
        self.model.fit(data_all.cpu())

    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        # Input of shape (B, T, D), output of shape (B,)

        # Extract time dimension
        t_dim = 1 if self.input_shape[0] == "b" else 0

        if self.first_diffs:  # Take fd over time dim
            X = torch.diff(inputs[0], dim=t_dim)
            # Since diff "deletes" first observation in batch
            # we dublicate first fd
            X = torch.cat(
                (X[:, 0:1, :] if t_dim == 1 else X[0:1, :, :], X),
                dim=t_dim,
            )
        else:
            X = inputs[0]

        # Transform data to latent
        X_hat = self.model.transform(X.view(X.shape[0] * X.shape[1], -1).cpu())
        X_hat = torch.from_numpy(self.model.inverse_transform(X_hat)).float()
        X_hat = X_hat.reshape_as(X).to(self.dummy.device)

        # Calculate scores: Subtract mean and multiply with inverse std
        scores = (X - X_hat).pow(2)

        if self.cum_method == "max":
            scores = scores.max(dim=-1).values
        else:
            scores = scores.mean(dim=-1)

        return scores[:, -1] if t_dim == 1 else scores[-1, :]

    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        # Input of shape (B, T) or (T, B), output of shape (B)
        (target,) = targets

        # Just return the last label of the window
        return target[:, -1] if self.input_shape[0] == "b" else target[-1]


class KernelPCAAnomalyDetector(PCAAnomalyDetector):
    def __init__(
        self,
        n_components: int,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        first_diffs: bool = False,
        cum_method: str = "mean",
        input_shape: str = "btf",
        **kwargs,
    ) -> None:
        """
        KernelPCA Anomaly Detector
            A simlpe anomaly detector that uses Kernel PCA as reconstruction method.

        Args:
            n_components[int, float]: number of principle components to keep. If float with 0<n_comp<=1.,the number
                of components will equal the explained variance. Defaults equals 0.95.
            first_diffs[bool]: Flag, if instead of raw values first difference should be used. Default is False.
            cum_method[str]: One of {'mean', 'max'}. Accumulation method over feature dimension. Note that when
                `feature_index` is not None. Then accumulation method will be ignored. Default is 'mean'.
        """
        super(KernelPCAAnomalyDetector, self).__init__(
            n_components, first_diffs, cum_method, input_shape
        )
        self.model = KernelPCA(
            n_components, kernel, gamma, fit_inverse_transform=True, **kwargs
        )
