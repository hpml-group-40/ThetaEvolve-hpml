import unittest

import numpy as np

from examples.anomaly_detection.evaluators.verify import compute_metrics, validate_solution


class VerifyTests(unittest.TestCase):
    def test_metrics_pool_counts_across_kpis(self):
        reference = {
            "missing": {
                "small": np.array([False]),
                "large": np.zeros(9, dtype=bool),
            },
            "labels": {
                "small": np.array([1], dtype=bool),
                "large": np.zeros(9, dtype=bool),
            },
        }
        predictions = {"small": [True], "large": [True] + [False] * 8}

        metrics = compute_metrics(predictions, reference)

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_missing_points_are_excluded(self):
        reference = {
            "missing": {"kpi": np.array([True, False, False])},
            "labels": {"kpi": np.array([True, True, False])},
        }

        metrics = compute_metrics({"kpi": [False, True, True]}, reference)

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_all_negative_and_all_positive_cases(self):
        negative_reference = {
            "missing": {"kpi": np.zeros(3, dtype=bool)},
            "labels": {"kpi": np.zeros(3, dtype=bool)},
        }
        positive_reference = {
            "missing": {"kpi": np.zeros(3, dtype=bool)},
            "labels": {"kpi": np.ones(3, dtype=bool)},
        }

        zero = compute_metrics({"kpi": [False, False, False]}, negative_reference)
        perfect = compute_metrics({"kpi": [True, True, True]}, positive_reference)

        self.assertEqual(zero, {"f1": 0.0, "precision": 0.0, "recall": 0.0})
        self.assertEqual(perfect, {"f1": 1.0, "precision": 1.0, "recall": 1.0})

    def test_rejects_mismatched_keys_lengths_and_values(self):
        reference = {
            "missing": {"kpi": np.zeros(2, dtype=bool)},
            "labels": {"kpi": np.zeros(2, dtype=bool)},
        }

        self.assertFalse(validate_solution({"other": [False, False]}, reference)[0])
        self.assertFalse(validate_solution({"kpi": [False]}, reference)[0])
        self.assertFalse(validate_solution({"kpi": [False, 2]}, reference)[0])


if __name__ == "__main__":
    unittest.main()