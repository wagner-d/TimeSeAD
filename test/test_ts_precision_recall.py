import os
import subprocess
import random
import sys
import time
from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from timesead.evaluation import constant_bias_fn, back_bias_fn, front_bias_fn, middle_bias_fn, \
    inverse_proportional_cardinality_fn, ts_precision_and_recall

# This is the path to the reference implementation
from timesead.evaluation.ts_precision_recall import improved_cardinality_fn

EVALUATOR_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'ts_recall_comparison',
                                               'reference_implementation'))
EXE = 'evaluate.exe' if sys.platform == 'win32' else 'evaluate'

BIAS_FN_MAP = {
    'flat': constant_bias_fn,
    'front': front_bias_fn,
    'middle': middle_bias_fn,
    'back': back_bias_fn
}

CARDINALITY_FN_MAP = {
    'reciprocal': inverse_proportional_cardinality_fn
}


class TestTSPrecisionAndRecall(TestCase):
    files = set()

    @classmethod
    def setUpClass(cls):
        src_dir = os.path.join(EVALUATOR_PATH, 'src')
        if not os.path.isfile(os.path.join(src_dir, EXE)):
            # Compile the reference implementation
            subprocess.run(f'g++ -fPIC -Wall -std=c++11 -O2 -g -o {EXE} main.cpp evaluator.cpp', shell=True,
                           cwd=src_dir)

        # Read all example files
        example_dir = os.path.join(EVALUATOR_PATH, 'examples')
        for subdir, dirs, files in os.walk(example_dir):
            real_files = [os.path.join(subdir, f[:-5]) for f in files if f.endswith('.real')]
            cls.files.update(real_files)
        cls.files = list(cls.files)

    @staticmethod
    def get_reference_results(file: str, alpha: float, cardinality: str, r_bias: str, p_bias: str):
        program = os.path.join(EVALUATOR_PATH, 'src', EXE)
        result = subprocess.run([program, '-t',  # We want the TS score
                                 file + '.real',    # First comes the ground-truth file
                                 file + '.pred',    # Then the prediction file
                                 '1',               # We don't care about the F-score, so beta does not matter
                                 str(alpha),        # Specify alpha
                                 cardinality,       # Cardinality function to use
                                 p_bias,            # And finally the bias functions for precision and recall
                                 r_bias],
                                capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError('Some Error occurred while running the reference implementation:', result.stderr)

        p_line, r_line, _, _ = result.stdout.splitlines(keepends=False)
        precision = float(p_line.split(' = ')[1])
        recall = float(r_line.split(' = ')[1])

        return precision, recall

    def test_test_class(self):
        file = self.files[0]
        pr, rec = self.get_reference_results(file, 0, 'reciprocal', 'flat', 'flat')

    @staticmethod
    def get_our_results(file: str, alpha: float, cardinality: str, r_bias: str, p_bias: str):
        real_labels = pd.read_csv(f'{file}.real', names=['data'])
        pred_labels = pd.read_csv(f'{file}.pred', names=['data'])

        real_labels = torch.from_numpy(real_labels['data'].to_numpy(dtype=np.int64))
        pred_labels = torch.from_numpy(pred_labels['data'].to_numpy(dtype=np.int64))

        return ts_precision_and_recall(real_labels, pred_labels, alpha,
                                       BIAS_FN_MAP[r_bias], CARDINALITY_FN_MAP[cardinality],
                                       BIAS_FN_MAP[p_bias], CARDINALITY_FN_MAP[cardinality])

    def test_simple_input(self):
        ground_truth = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        prediction = torch.tensor([0, 1, 1, 0, 0, 0, 1, 0, 0, 0])

        expected_precision = 0.25
        expected_recall = 0.5

        precision, recall = ts_precision_and_recall(ground_truth, prediction, alpha=0)
        self.assertEqual(expected_precision, precision)
        self.assertEqual(expected_recall, recall)

    def test_simple_improved(self):
        scores = torch.rand(1000)
        labels = torch.empty(1000, dtype=torch.long).bernoulli_(0.1)

        thresholds = torch.unique(input=scores, sorted=True)

        precision = []
        recall = []
        for t in thresholds:
            # prec, rec = ts_precision_and_recall(labels, (scores > t).int(), alpha=0)
            prec, rec = ts_precision_and_recall(labels, (scores > t).int(), alpha=0,
                                                recall_cardinality_fn=improved_cardinality_fn)
            precision.append(prec)
            recall.append(rec)

        self.assertTrue(all(recall[i] >= recall[i+1] for i in range(len(recall) - 1)))

    def test_improved_multiple(self):
        times = 100

        for _ in range(times):
            self.test_simple_improved()

    def test_simple_failure_case(self):
        ground_truth = torch.tensor([1, 1, 1, 1])
        scores = torch.tensor([1, 1, 0, 0.5])
        t_1 = 0
        t_2 = 0.5
        p_t_1 = (scores > t_1).long()
        p_t_2 = (scores > t_2).long()

        expected_recall_1 = 0.375
        expected_recall_2 = 0.5

        precision_1, recall_1 = ts_precision_and_recall(ground_truth, p_t_1, alpha=0)
        precision_2, recall_2 = ts_precision_and_recall(ground_truth, p_t_2, alpha=0)
        self.assertEqual(expected_recall_1, recall_1)
        self.assertEqual(expected_recall_2, recall_2)

    def test_failure_case_same_recall_different_precision(self):
        ground_truth = torch.tensor([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        scores = torch.tensor([1, 1, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0, 0, 0])
        t_1 = 0
        t_2 = 0.5
        p_t_1 = (scores > t_1).long()
        p_t_2 = (scores > t_2).long()

        expected_precision_1 = 0.7
        expected_recall_1 = 0.375
        expected_precision_2 = 5/6
        expected_recall_2 = 0.375

        precision_1, recall_1 = ts_precision_and_recall(ground_truth, p_t_1, alpha=0)
        precision_2, recall_2 = ts_precision_and_recall(ground_truth, p_t_2, alpha=0)
        self.assertEqual(expected_precision_1, precision_1)
        self.assertEqual(expected_recall_1, recall_1)
        self.assertAlmostEqual(expected_precision_2, precision_2)
        self.assertEqual(expected_recall_2, recall_2)

    def test_large_input(self):
        ground_truth = torch.empty((25000,), dtype=torch.int64).bernoulli_(0.2)
        prediction = torch.empty((25000,), dtype=torch.int64).bernoulli_(0.2)

        duration = float('inf')
        for _ in range(100):
            start = time.perf_counter()
            ts_precision_and_recall(ground_truth, prediction, alpha=0)
            end = time.perf_counter()
            duration = min(duration, end - start)

        print(f'Took {duration} seconds to compute.')

    def compare_on_all_files(self, alpha: float, cardinality: str, r_bias: str, p_bias: str):
        for file in self.files:
            reference_result = self.get_reference_results(file, alpha, cardinality, r_bias, p_bias)
            our_result = self.get_our_results(file, alpha, cardinality, r_bias, p_bias)
            # The reference implementation seems to round the output at some point
            self.assertTrue(np.allclose(reference_result, our_result))

    def test_alpha_zero(self):
        self.compare_on_all_files(0, 'reciprocal', 'flat', 'flat')

    def test_alpha_nonzero(self):
        self.compare_on_all_files(0.5, 'reciprocal', 'flat', 'flat')
        self.compare_on_all_files(1, 'reciprocal', 'flat', 'flat')
        self.compare_on_all_files(0.12345, 'reciprocal', 'flat', 'flat')

    def test_middle_bias(self):
        self.compare_on_all_files(0, 'reciprocal', 'middle', 'flat')
        self.compare_on_all_files(0, 'reciprocal', 'flat', 'middle')
        self.compare_on_all_files(0, 'reciprocal', 'middle', 'middle')
        self.compare_on_all_files(0.75, 'reciprocal', 'middle', 'flat')
        self.compare_on_all_files(1, 'reciprocal', 'flat', 'middle')
        self.compare_on_all_files(0.5, 'reciprocal', 'middle', 'middle')

    def run_random_test(self):
        alpha = random.random()
        cardinality = random.choice(list(CARDINALITY_FN_MAP.keys()))
        r_bias = random.choice(list(BIAS_FN_MAP.keys()))
        p_bias = random.choice(list(BIAS_FN_MAP.keys()))

        print('Parameters:', alpha, cardinality, r_bias, p_bias)

        self.compare_on_all_files(alpha, cardinality, r_bias, p_bias)

    def test_random_config(self):
        for _ in range(100):
            self.run_random_test()