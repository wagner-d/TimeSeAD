from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn.functional as F

from ..data.dataset import BaseTSDataset


def get_data_all(dataset: torch.utils.data.Dataset, take_fd: bool = False,
) -> torch.Tensor:
    """Extract data matrix from dataset.

    :param dataset: Time-series dataset.
           take_fd: flag, to take first differences.
    :type dataset: torch.utils.data.Dataset
          take_id: bool
    :return: data matrix
    :rtype: torch.Tensor
    """
    data_all = ()
    for i in range(len(dataset)):
        inputs = dataset[i][0][0]
        # Take first differences
        if take_fd:
            inputs = torch.diff(inputs, dim=0)

        data_all += (inputs,)

    # Combine dataset to one big matrix
    return torch.cat(data_all)

def compute_whiten_matrix(
    dataset: torch.utils.data.Dataset, take_fd: bool = False,
) -> torch.Tensor:
    """Compute whiten matrix of dataset.

    :param dataset: Time-series dataset.
    :type dataset: torch.utils.data.Dataset
    :return: whiten matrix
    :rtype: torch.Tensor
    """

    data_all = get_data_all(dataset, take_fd)
    # Demean data
    data_all -= data_all.mean(0)
    # Calc covariance
    Sigma = data_all.T @ data_all / (data_all.shape[0] - 1)
    # Calculating Eigenvalues and Eigenvectors of the covariance matrix
    Lamb, V = torch.linalg.eigh(Sigma)
    # Deal with multiple constant features
    Lamb[Lamb < torch.finfo(torch.float32).eps] = 1
    # ZCA Whitening Matrix
    return V @ torch.diag(1./torch.sqrt(Lamb)) @ V.T

def compute_feature_statistics(
    dataset: torch.utils.data.Dataset, take_fd: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Compute statistics of each feature.

    :param dataset: Time-series dataset.
    :type dataset: torch.utils.data.Dataset
    :return: Normal mean, normal std, anomaly mean, anomaly std, minimum, and maximum
    :rtype: Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]
    """

    minimum = np.inf * torch.ones(1, dataset[0][0][0].shape[1])
    maximum = -np.inf * torch.ones(1, dataset[0][0][0].shape[1])

    for i in range(len(dataset)):

        inputs = dataset[i][0][0]

        # Take first differences
        if take_fd:
            inputs = torch.diff(inputs, dim=0)

        minimum = torch.cat((inputs.min(0, keepdim=True).values, minimum))
        maximum = torch.cat((inputs.max(0, keepdim=True).values, maximum))

    minimum = minimum.min(0).values
    maximum = maximum.max(0).values

    normal_mean, normal_std, anomaly_mean, anomaly_std = compute_feature_mean_std(
        dataset, take_fd
    )

    return normal_mean, normal_std, anomaly_mean, anomaly_std, minimum, maximum


def compute_feature_mean_std(
    dataset: BaseTSDataset, take_fd: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Compute the mean and standard deviation of each feature.
    :param dataset: Time-series dataset.
    :type dataset: torch.utils.data.Dataset
    :return: Normal mean, normal std, anomaly mean, anomaly std
    :rtype: Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
    """

    normal_mean = torch.zeros(dataset.num_features)
    anomaly_mean = torch.zeros(dataset.num_features)

    n_normal_points = 0
    n_anomaly_points = 0

    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        # Shape (T, D_1, ..., D_n)
        inputs = inputs[0]
        # Shape (T,)
        targets = targets[0]

        # Take first differences
        if take_fd:
            inputs = torch.diff(inputs, dim=0)
            targets = targets[1:]

        normal_points = inputs[targets == 0]
        anomaly_points = inputs[targets != 0]

        n_normal_points += len(normal_points)
        n_anomaly_points += len(anomaly_points)

        if len(normal_points) > 0:
            normal_mean += normal_points.sum(dim=0)
        if len(anomaly_points) > 0:
            anomaly_mean += anomaly_points.sum(dim=0)

    normal_mean /= n_normal_points
    anomaly_mean /= n_anomaly_points

    normal_std = torch.zeros(dataset.num_features)
    anomaly_std = torch.zeros(dataset.num_features)

    # Compute std in the second pass
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        inputs = inputs[0]
        targets = targets[0]

        # Take first differences
        if take_fd:
            inputs = torch.diff(inputs, dim=0)
            targets = targets[1:]

        normal_points = inputs[targets == 0]
        anomaly_points = inputs[targets != 0]

        if len(normal_points) > 0:
            normal_std += torch.sum((normal_points - normal_mean).pow(2), dim=0)
        if len(anomaly_points) > 0:
            anomaly_std += torch.sum((anomaly_points - anomaly_mean).pow(2), dim=0)

    normal_std = torch.sqrt(normal_std / (n_normal_points - 1))
    anomaly_std = torch.sqrt(anomaly_std / (n_anomaly_points - 1))

    if n_normal_points == 0:
        normal_mean, normal_std = None, None

    if n_anomaly_points == 0:
        anomaly_mean, anomaly_std = None, None

    return normal_mean, normal_std, anomaly_mean, anomaly_std


def compute_ts_statisitcs(
    timeseries: torch.Tensor, targets: torch.Tensor, dim_order: str = "tf"
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:

    data = timeseries.permute((dim_order.index("t"), dim_order.index("f")))

    normal_mean = (
        torch.mean(data[targets == 0], 0)
        if torch.sum(targets) < targets.shape[0]
        else None
    )
    normal_std = (
        torch.std(data[targets == 0], 0)
        if torch.sum(targets) < targets.shape[0]
        else None
    )
    anomaly_mean = torch.mean(data[targets == 1], 0) if torch.sum(targets) > 0 else None
    anomaly_std = torch.std(data[targets == 1], 0) if torch.sum(targets) > 0 else None
    minimum = torch.min(data, 0).values
    maximum = torch.max(data, 0).values

    return normal_mean, normal_std, anomaly_mean, anomaly_std, minimum, maximum


def training_means(dataset):

    stats = compute_feature_statistics(dataset)

    return stats[0], stats[1]


def compute_anomaly_positions(dataset: torch.utils.data.Dataset) -> List[int]:
    """Computes the positions of anomalies in the dataset.

    :param dataset: Dataset to compute the statistics of.
    :type dataset: torch.utils.data.Dataset
    :return: List of all the relative positions of anomalies in the dataset.
    :rtype: List[int]
    """

    positions = []

    for i in range(len(dataset)):

        _, targets = dataset[i]
        targets = targets[0]

        positions.extend(
            ((targets == 1).nonzero(as_tuple=True)[0] / targets.shape[0]).tolist()
        )

    return positions


def compute_anomaly_lengths(dataset: torch.utils.data.Dataset) -> List[int]:
    """Computes the length of each anomalous window in the dataset.

    :param dataset: Dataset to compute the statistics of.
    :type dataset: torch.utils.data.Dataset
    :return: List of lengths of anomalies in the dataset.
    :rtype: List[int]
    """

    lengths = []

    for i in range(len(dataset)):

        _, targets = dataset[i]
        targets = targets[0]
        targets = 2 * targets - F.pad(
            input=targets[:-1], pad=(1, 0), mode="constant", value=0
        )
        targets = targets[targets > 0]
        indices = torch.where(targets > 1)[0].tolist()

        indices.append(targets.shape[0])

        for j, k in zip(indices[:-1], indices[1:]):
            lengths.append(k - j)

    return lengths


def compute_total_time_steps(dataset: torch.utils.data.Dataset) -> int:
    """Compute the total amount of time steps in the dataset (normal + anormal)

    :param dataset: Dataset to compute the statistics of.
    :type dataset: torch.utils.data.Dataset
    :return: Number of time steps in the dataset
    :rtype: int
    """

    n = 0

    for i in range(len(dataset)):

        _, targets = dataset[i]
        n += targets[0].shape[0]

    return n
