from typing import Callable, Tuple, Dict, Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

from .ts_precision_recall import constant_bias_fn, inverse_proportional_cardinality_fn, ts_precision_and_recall, \
    improved_cardinality_fn, compute_window_indices


class Evaluator:
    """
    A class that can compute several evaluation metrics for a dataset. Each method returns the score as a single float,
    but it can also return additional information in a dict.
    """

    def auc(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the classic point-wise area under the receiver operating characteristic curve.
        
        This will return a value between 0 and 1 where 1 indicates a perfect classifier.

        .. seealso::
            Scikit-learns's :func:`~sklearn.metrics.roc_auc_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :return: A tuple consisting of the AUC score and an empty dict.
        """

        return roc_auc_score(labels.numpy(), scores.numpy()), {}

    def f1_score(self, labels: torch.Tensor, scores: torch.Tensor, pos_label: int = 1) -> Tuple[float, Dict[str, Any]]:
        """Compute the classic point-wise F1 score.
        
        This will return a value between 0 and 1 where 1 indicates a perfect classifier.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.f1_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing binary predictions of whether a point is an anomaly or
            not.
        :param pos_label: Class to report.
        :return: A tuple consisting of the F1 score and an empty dict.
        """

        return f1_score(labels.numpy(), scores.numpy(), pos_label=pos_label).item(), {}

    def best_fbeta_score(self, labels: torch.Tensor, scores: torch.Tensor, beta: float) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the classic point-wise :math:`F_{\beta}` score.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{\beta}`
        score for the resulting binary predictions. It then returns the highest score.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.fbeta_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :param beta: Positive number that determines the trade-off between precision and recall when computing the
            F-score. :math:`\beta = 1` assigns equal weight to both while :math:`\beta < 1` emphasizes precision and
            vice versa.
        :return: A tuple consisting of the best :math:`F_{\beta}` score and a dict containing the threshold that
            produced the maximal score.
        """
        precision, recall, thresholds = precision_recall_curve(labels.numpy(), scores.numpy())

        f_score = np.nan_to_num((1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall), nan=0)
        best_index = np.argmax(f_score)

        return f_score[best_index].item(), dict(threshold=thresholds[best_index].item())

    def best_f1_score(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the classic point-wise :math:`F_{1}` score.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{1}`
        score for the resulting binary predictions. It then returns the highest score.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.f1_score` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :return: A tuple consisting of the best :math:`F_{1}` score and a dict containing the threshold that
            produced the maximal score.
        """

        return self.best_fbeta_score(labels, scores, 1)

    def auprc(self, labels: torch.Tensor, scores: torch.Tensor, integration: str = 'trapezoid') -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the classic point-wise area under the precision-recall curve.

        This will return a value between 0 and 1 where 1 indicates a perfect classifier.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.average_precision` function.

            Scikit-learn's :func:`~sklearn.metrics.precision_recall_curve` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`.
        :param integration: Method to use for computing the area under the curve. ``'riemann'`` corresponds to a simple
            Riemann sum, whereas ``'trapezoid'`` uses the trapezoidal rule.
        :return: A tuple consisting of the AuPRC score and an empty dict.
        """
        precision, recall, thresholds = precision_recall_curve(labels.numpy(), scores.numpy())
        # recall is nan in the case where all ground-truth labels are 0. Simply set it to zero here
        # so that it does not contribute to the area
        recall = np.nan_to_num(recall, nan=0)

        if integration == 'riemann':
            area = -np.sum(np.diff(recall) * precision[:-1])
        else:
            area = auc(recall, precision)

        return area.item(), {}

    def average_precision(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the classic point-wise average precision score.

        .. note::
            This is just a shorthand for :meth:`auprc` with ``integration='riemann'``.

        .. seealso::
            Scikit-learn's :func:`~sklearn.metrics.average_precision` function.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the average precision score and an empty dict.
        """
        return self.auprc(labels, scores, integration='riemann')

    def ts_auprc(self, labels: torch.Tensor, scores: torch.Tensor, integration='trapezoid',
                 weighted_precision: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the area under the precision-recall curve using precision and recall for time series [Tatbul2018]_.

        .. note::
            This function uses the improved cardinality function described in [Wagner2023]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param integration: Method to use for computing the area under the curve. ``'riemann'`` corresponds to a simple
            Riemann sum, whereas ``'trapezoid'`` uses the trapezoidal rule.
        :param weighted_precision: If ``True``, the precision score of a predicted window will be weighted with the
            length of the window in the final score. Otherwise, each window will have the same weight.
        :return: A tuple consisting of the AuPRC score and an empty dict.

        .. [Tatbul2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich.
            Precision and recall for time series. Advances in neural information processing systems. 2018;31.
        .. [Wagner2023] D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
            TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
            Transactions on Machine Learning Research (TMLR), (to appear) 2023.
        """
        thresholds = torch.unique(input=scores, sorted=True)

        precision = torch.empty(thresholds.shape[0] + 1, dtype=torch.float, device=thresholds.device)
        recall = torch.empty(thresholds.shape[0] + 1, dtype=torch.float, device=thresholds.device)
        predictions = torch.empty_like(scores, dtype=torch.long)

        # Set last values when threshold is at infinity so that no point is predicted as anomalous.
        # Precision is not defined in this case, we set it to 1 to stay consistent with scikit-learn
        precision[-1] = 1
        recall[-1] = 0

        label_ranges = compute_window_indices(labels)

        for i, t in enumerate(thresholds):
            torch.greater_equal(scores, t, out=predictions)
            prec, rec = ts_precision_and_recall(labels, predictions, alpha=0,
                                                recall_cardinality_fn=improved_cardinality_fn,
                                                anomaly_ranges=label_ranges,
                                                weighted_precision=weighted_precision)
            precision[i] = prec
            recall[i] = rec

        if integration == 'riemann':
            area = -torch.sum(torch.diff(recall) * precision[:-1])
        else:
            area = auc(recall.numpy(), precision.numpy())

        return area.item(), {}

    def ts_average_precision(self, labels: torch.Tensor, scores: torch.Tensor, weighted_precision: bool = True) \
            -> Tuple[float, Dict[str, Any]]:
        """
        Compute the average precision score using precision and recall for time series [Tatbul2018]_.

        .. note::
            This is just a shorthand for :meth:`ts_auprc` with ``integration='riemann'``.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param weighted_precision: If ``True``, the precision score of a predicted window will be weighted with the
            length of the window in the final score. Otherwise, each window will have the same weight.
        :return: A tuple consisting of the average precision score and an empty dict.
        """

        return self.ts_auprc(labels, scores, integration='riemann', weighted_precision=weighted_precision)

    def ts_auprc_unweighted(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the area under the precision-recall curve using precision and recall for time series [Tatbul2018]_.

        .. note::
            This is just a shorthand for :meth:`ts_auprc` with ``integration='riemann'`` and
            ``weighted_precision=False``.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the AuPRC score and an empty dict.
        """
        return self.ts_auprc(labels, scores, integration='trapezoid', weighted_precision=False)

    def __best_ts_fbeta_score(self, labels: torch.Tensor, scores: torch.Tensor, beta: float,
                              recall_cardinality_fn: Callable = improved_cardinality_fn,
                              weighted_precision: bool = True) -> Tuple[float, Dict[str, Any]]:
        thresholds = torch.unique(input=scores, sorted=True)

        precision = torch.empty_like(thresholds, dtype=torch.float)
        recall = torch.empty_like(thresholds, dtype=torch.float)
        predictions = torch.empty_like(scores, dtype=torch.long)

        label_ranges = compute_window_indices(labels)
        # label_ranges = None

        for i, t in enumerate(thresholds):
            torch.greater(scores, t, out=predictions)
            prec, rec = ts_precision_and_recall(labels, predictions, alpha=0,
                                                recall_cardinality_fn=recall_cardinality_fn,
                                                anomaly_ranges=label_ranges,
                                                weighted_precision=weighted_precision)

            # We need to handle the case where precision and recall are both 0. This can either happen for an
            # extremely bad classifier or if all predictions are 0
            if prec == rec == 0:
                # We simply set rec = 1 to avoid dividing by zero. The F-score will still be 0
                rec = 1

            precision[i] = prec
            recall[i] = rec

        f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        max_score_index = torch.argmax(f_score)

        return f_score[max_score_index].item(), dict(threshold=thresholds[max_score_index].item(),
                                                     precision=precision[max_score_index].item(),
                                                     recall=recall[max_score_index].item())

    def best_ts_fbeta_score(self, labels: torch.Tensor, scores: torch.Tensor, beta: float) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the :math:`F_{\beta}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{\beta}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the improved cardinality function and weighted precision as described in [Wagner2023]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param beta: Positive number that determines the trade-off between precision and recall when computing the
            F-score. :math:`\beta = 1` assigns equal weight to both while :math:`\beta < 1` emphasizes precision and
            vice versa.
        :return: A tuple consisting of the best :math:`F_{\beta}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.__best_ts_fbeta_score(labels, scores, beta, recall_cardinality_fn=improved_cardinality_fn,
                                        weighted_precision=True)

    def best_ts_fbeta_score_classic(self, labels: torch.Tensor, scores: torch.Tensor, beta: float) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the :math:`F_{\beta}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{\beta}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the default cardinality function (:math:`\frac[1}{x}`) and unweighted precision, i.e.,
            the default parameters described in [Tatbul2018]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :param beta: Positive number that determines the trade-off between precision and recall when computing the
            F-score. :math:`\beta = 1` assigns equal weight to both while :math:`\beta < 1` emphasizes precision and
            vice versa.
        :return: A tuple consisting of the best :math:`F_{\beta}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.__best_ts_fbeta_score(labels, scores, beta,
                                          recall_cardinality_fn=inverse_proportional_cardinality_fn,
                                          weighted_precision=False)

    def best_ts_f1_score(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the :math:`F_{1}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{1}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the improved cardinality function and weighted precision as described in [Wagner2023]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the best :math:`F_{1}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.best_ts_fbeta_score(labels, scores, 1)

    def best_ts_f1_score_classic(self, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        r"""
        Compute the :math:`F_{1}` score using precision and recall for time series [Tatbul2018]_.

        This method will apply all possible thresholds to the values in ``scores`` and compute the :math:`F_{1}`
        score for the resulting binary predictions. It then returns the highest score.

        .. note::
            This function uses the default cardinality function (:math:`\frac[1}{x}`) and unweighted precision, i.e.,
            the default parameters described in [Tatbul2018]_.

        :param labels: A 1-D :class:`~torch.Tensor` containing the ground-truth labels. 1 corresponds to an anomaly,
            0 means that the point is normal.
        :param scores: A 1-D :class:`~torch.Tensor` containing the scores returned by an
            :class:`~timesead.models.common.AnomalyDetector`
        :return: A tuple consisting of the best :math:`F_{1}` score and a dict containing the threshold, recall and
            precision that produced the maximal score.
        """
        return self.best_ts_fbeta_score_classic(labels, scores, 1)
