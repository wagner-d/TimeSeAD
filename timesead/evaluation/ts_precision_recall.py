from typing import List, Tuple, Callable, Optional

import torch


def constant_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a constant bias function that assigns the same weight to all positions.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{1}{n} \sum_{i = 1}^{n} \text{inputs}_i,

    where :math:`n = \lvert \text{inputs} \rvert`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    return torch.sum(inputs).item() / inputs.shape[0]


def back_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a bias function that assigns the more weight to predictions towards the back of a
    ground-truth anomaly window.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{2}{n * (n + 1)} \sum_{i = 1}^{n} \text{inputs}_i \cdot i,

    where :math:`n = \lvert \text{inputs} \rvert`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    n = inputs.shape[0]
    res = torch.dot(inputs, torch.arange(1, n + 1, dtype=inputs.dtype, device=inputs.device)).item()
    res /= (n * (n + 1)) // 2  # sum of numbers 1, ..., n
    return res


def front_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a bias function that assigns the more weight to predictions towards the front of a
    ground-truth anomaly window.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{2}{n * (n + 1)} \sum_{i = 1}^{n} \text{inputs}_i \cdot (n + 1 - i),

    where :math:`n = \lvert \text{inputs} \rvert`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    n = inputs.shape[0]
    res = torch.dot(inputs, torch.arange(n, 0, -1, dtype=inputs.dtype, device=inputs.device)).item()
    res /= (n * (n + 1)) // 2  # sum of numbers 1, ..., n
    return res


def middle_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a bias function that assigns the more weight to predictions in the middle of a
    ground-truth anomaly window.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{2}{m * (m + 1) + (n - m) * (n - m + 1)} \sum_{i = 1}^{n} \text{inputs}_i \cdot
        \begin{cases}
            i & \text{if } i \leq m\\
            (n + 1 - i) & \text{otherwise}
        \end{cases},

    where :math:`n = \lvert \text{inputs} \rvert` and :math:`m = \lceil \frac{n}{2} \rceil`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    n = inputs.shape[0]
    result = torch.empty_like(inputs)
    middle, remainder = divmod(n, 2)
    middle2 = middle + remainder
    torch.arange(1, middle + 1, out=result[:middle], dtype=result.dtype, device=result.device)
    torch.arange(middle2, 0, -1, out=result[-middle2:], dtype=result.dtype, device=result.device)
    result = torch.dot(inputs, result).item()
    result /= (middle * (middle + 1) + middle2 * (middle2 + 1)) // 2
    return result


def inverse_proportional_cardinality_fn(cardinality: int, gt_length: int) -> float:
    r"""
    Cardinality function that assigns an inversely proportional weight to predictions within a single ground-truth
    window.

    This is the default cardinality function recommended in [Tatbul2018]_.

    .. note::
       This function leads to a metric that is not recall-consistent! Please see [Wagner2023]_ for more details.

    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window (unused).
    :return: The cardinality factor :math:`\frac{1}{\text{cardinality}}`.

    .. [Tatbul2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich.
        Precision and recall for time series. Advances in neural information processing systems. 2018;31.
    .. [Wagner2023] D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
        TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
        Transactions on Machine Learning Research (TMLR), (to appear) 2023.
    """
    return 1 / max(1, cardinality)


def improved_cardinality_fn(cardinality: int, gt_length: int):
    r"""
    Recall-consistent cardinality function introduced by [Wagner2023]_ that assigns lower weight to ground-truth windows
    that overlap with many predicted windows.

    This function computes

    .. math::
        \left(\frac{\text{gt_length} - 1}{\text{gt_length}}\right)^{\text{cardinality} - 1}.

    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window.
    :return: The cardinality factor.
    """
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)


def compute_window_indices(binary_labels: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Compute a list of indices where anomaly windows begin and end.

    :param binary_labels: A 1-D :class:`~torch.Tensor` containing ``1`` for an anomalous time step or ``0`` otherwise.
    :return: A list of tuples ``(start, end)`` for each anomaly window in ``binary_labels``, where ``start`` is the
        index at which the window starts and ``end`` is the first index after the end of the window.
    """
    boundaries = torch.empty_like(binary_labels)
    boundaries[0] = 0
    boundaries[1:] = binary_labels[:-1]
    boundaries *= -1
    boundaries += binary_labels
    # boundaries will be 1 where a window starts and -1 at the end of a window

    indices = torch.nonzero(boundaries, as_tuple=True)[0].tolist()
    if len(indices) % 2 != 0:
        # Add the last index as the end of a window if appropriate
        indices.append(binary_labels.shape[0])
    indices = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]

    return indices


def _compute_overlap(preds: torch.Tensor, pred_indices: List[Tuple[int, int]],
                     gt_indices: List[Tuple[int, int]], alpha: float,
                     bias_fn: Callable, cardinality_fn: Callable,
                     use_window_weight: bool = False) -> float:
    n_gt_windows = len(gt_indices)
    n_pred_windows = len(pred_indices)
    total_score = 0.0
    total_gt_points = 0

    i = j = 0
    while i < n_gt_windows and j < n_pred_windows:
        gt_start, gt_end = gt_indices[i]
        window_length = gt_end - gt_start
        total_gt_points += window_length
        i += 1

        cardinality = 0
        while j < n_pred_windows and pred_indices[j][1] <= gt_start:
            j += 1
        while j < n_pred_windows and pred_indices[j][0] < gt_end:
            j += 1
            cardinality += 1

        if cardinality == 0:
            # cardinality == 0 means no overlap at all, hence no contribution
            continue

        # The last predicted window that overlaps our current window could also overlap the next window.
        # Therefore, we must consider it again in the next loop iteration.
        j -= 1

        cardinality_multiplier = cardinality_fn(cardinality, window_length)

        prediction_inside_ground_truth = preds[gt_start:gt_end]
        # We calculate omega directly in the bias function, because this can greatly improve running time
        # for the constant bias, for example.
        omega = bias_fn(prediction_inside_ground_truth)

        # Either weight evenly across all windows or based on window length
        weight = window_length if use_window_weight else 1

        # Existence reward (if cardinality > 0 then this is certainly 1)
        total_score += alpha * weight
        # Overlap reward
        total_score += (1 - alpha) * cardinality_multiplier * omega * weight

    denom = total_gt_points if use_window_weight else n_gt_windows

    return total_score / denom


def ts_precision_and_recall(anomalies: torch.Tensor, predictions: torch.Tensor, alpha: float = 0,
                            recall_bias_fn: Callable[[torch.Tensor], float] = constant_bias_fn,
                            recall_cardinality_fn: Callable[[int], float] = inverse_proportional_cardinality_fn,
                            precision_bias_fn: Optional[Callable] = None,
                            precision_cardinality_fn: Optional[Callable] = None,
                            anomaly_ranges: Optional[List[Tuple[int, int]]] = None,
                            prediction_ranges: Optional[List[Tuple[int, int]]] = None,
                            weighted_precision: bool = False) -> Tuple[float, float]:
    """
    Computes precision and recall for time series as defined in [Tatbul2018]_.

    .. note::
       The default parameters for this function correspond to the defaults recommended in [Tatbul2018]_. However,
       those might not be desirable in most cases, please see [Wagner2023]_ for a detailed discussion.

    :param anomalies: Binary 1-D :class:`~torch.Tensor` of shape ``(length,)`` containing the true labels.
    :param predictions: Binary 1-D :class:`~torch.Tensor` of shape ``(length,)`` containing the predicted labels.
    :param alpha: Weight for existence term in recall.
    :param recall_bias_fn: Function that computes the bias term for a given ground-truth window.
    :param recall_cardinality_fn: Function that compute the cardinality factor for a given ground-truth window.
    :param precision_bias_fn: Function that computes the bias term for a given predicted window.
        If ``None``, this will be the same as ``recall_bias_function``.
    :param precision_cardinality_fn: Function that computes the cardinality factor for a given predicted window.
        If ``None``, this will be the same as ``recall_cardinality_function``.
    :param weighted_precision: If True, the precision score of a predicted window will be weighted with the
        length of the window in the final score. Otherwise, each window will have the same weight.
    :param anomaly_ranges: A list of tuples ``(start, end)`` for each anomaly window in ``anomalies``, where ``start``
        is the index at which the window starts and ``end`` is the first index after the end of the window. This can
        be ``None``, in which case the list is computed automatically from ``anomalies``.
    :param prediction_ranges: A list of tuples ``(start, end)`` for each anomaly window in ``predictions``, where
        ``start`` is the index at which the window starts and ``end`` is the first index after the end of the window.
        This can be ``None``, in which case the list is computed automatically from ``predictions``.
    :return: A tuple consisting of the time-series precision and recall for the given labels.
    """
    has_anomalies = torch.any(anomalies > 0).item()
    has_predictions = torch.any(predictions > 0).item()

    # Catch special cases which would cause a division by zero
    if not has_predictions and not has_anomalies:
        # In this case, the classifier is perfect, so it makes sense to set precision and recall to 1
        return 1, 1
    elif not has_predictions or not has_anomalies:
        return 0, 0

    # Set precision functions to the same as recall functions if they are not given
    if precision_bias_fn is None:
        precision_bias_fn = recall_bias_fn
    if precision_cardinality_fn is None:
        precision_cardinality_fn = recall_cardinality_fn

    if anomaly_ranges is None:
        anomaly_ranges = compute_window_indices(anomalies)
    if prediction_ranges is None:
        prediction_ranges = compute_window_indices(predictions)

    recall = _compute_overlap(predictions, prediction_ranges, anomaly_ranges, alpha, recall_bias_fn,
                              recall_cardinality_fn)
    precision = _compute_overlap(anomalies, anomaly_ranges, prediction_ranges, 0, precision_bias_fn,
                                 precision_cardinality_fn, use_window_weight=weighted_precision)

    return precision, recall
