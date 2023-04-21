"""
This package contains functions for evaluating the performance of time-series anomaly detectors.

At the moment, the :class:`Evaluator` class supports classic point-wise metrics such as :math:`F_1`-score, area under
the precision recall curve, etc., and composite metrics derived from precision and recall for time series.
"""
from .evaluator import Evaluator
from .ts_precision_recall import ts_precision_and_recall, constant_bias_fn, back_bias_fn, front_bias_fn, middle_bias_fn,\
    inverse_proportional_cardinality_fn
