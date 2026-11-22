import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.anomaly_detection.score_heldout import score_heldout_data


class HeldoutScoringTests(unittest.TestCase):
    def test_scores_per_kpi_and_pooled_excluding_missing_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "kpi_heldout.npz"
            np.savez(
                data_path,
                small_error=np.array([0.0]),
                small_missing=np.array([False]),
                small_label=np.array([1], dtype=np.int8),
                large_error=np.zeros(10),
                large_missing=np.array([True] + [False] * 9),
                large_label=np.zeros(10, dtype=np.int8),
            )

            def detector(error, missing):
                return np.array([True]) if len(error) == 1 else np.array([False] * 10)

            per_kpi, pooled = score_heldout_data(detector, data_path)

        self.assertEqual(per_kpi["small"]["f1"], 1.0)
        self.assertEqual(per_kpi["large"]["f1"], 0.0)
        self.assertAlmostEqual(pooled["f1"], 1.0)

    def test_rejects_misaligned_heldout_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "kpi_heldout.npz"
            np.savez(
                data_path,
                kpi_error=np.zeros(2),
                kpi_missing=np.zeros(2, dtype=bool),
                kpi_label=np.zeros(1, dtype=np.int8),
            )

            with self.assertRaisesRegex(ValueError, "matching lengths"):
                score_heldout_data(lambda error, missing: np.zeros(len(error)), data_path)


if __name__ == "__main__":
    unittest.main()