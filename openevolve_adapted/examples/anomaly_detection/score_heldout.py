"""Score an evolved anomaly detector once against a labeled held-out archive."""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np


OPENEVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(OPENEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENEVOLVE_ROOT))

from examples.anomaly_detection.evaluators.verify import compute_metrics


def load_detector(program_path: Path) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Import a candidate file and return its detector without running its main."""
    spec = importlib.util.spec_from_file_location("heldout_anomaly_candidate", program_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load candidate program: {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detector = getattr(module, "detect_anomalies", None)
    if not callable(detector):
        raise ValueError(f"Candidate does not define callable detect_anomalies: {program_path}")
    return detector


def score_heldout_data(
    detector: Callable[[np.ndarray, np.ndarray], np.ndarray], data_path: Path
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Return per-KPI and pooled metrics using labels embedded in the held-out archive."""
    predictions = {}
    missing_by_kpi = {}
    labels_by_kpi = {}

    with np.load(data_path, allow_pickle=False) as archive:
        kpi_ids = sorted(
            key[: -len("_error")] for key in archive.files if key.endswith("_error")
        )
        label_ids = sorted(
            key[: -len("_label")] for key in archive.files if key.endswith("_label")
        )
        if not kpi_ids or kpi_ids != label_ids:
            raise ValueError("Held-out error and label KPI IDs must match and be non-empty")

        for kpi_id in kpi_ids:
            error = archive[f"{kpi_id}_error"]
            missing = archive[f"{kpi_id}_missing"]
            labels = archive[f"{kpi_id}_label"]
            if error.ndim != 1 or missing.ndim != 1 or labels.ndim != 1:
                raise ValueError(f"Held-out arrays must be 1-D for KPI {kpi_id}")
            if len(error) != len(missing) or len(error) != len(labels):
                raise ValueError(f"Held-out arrays must have matching lengths for KPI {kpi_id}")
            if missing.dtype.kind != "b":
                raise ValueError(f"Missing mask must be boolean for KPI {kpi_id}")
            if not np.isfinite(error).all():
                raise ValueError(f"Errors contain non-finite values for KPI {kpi_id}")
            if not np.isin(labels, (0, 1)).all():
                raise ValueError(f"Labels must contain only 0 or 1 for KPI {kpi_id}")

            prediction = np.asarray(detector(error.copy(), missing.copy()))
            if prediction.ndim != 1 or len(prediction) != len(error):
                raise ValueError(f"Detector must return one flag per point for KPI {kpi_id}")

            predictions[kpi_id] = prediction
            missing_by_kpi[kpi_id] = missing
            labels_by_kpi[kpi_id] = labels.astype(bool, copy=False)

    reference_data = {"missing": missing_by_kpi, "labels": labels_by_kpi}
    per_kpi_metrics = {
        kpi_id: compute_metrics(
            {kpi_id: predictions[kpi_id]},
            {
                "missing": {kpi_id: missing_by_kpi[kpi_id]},
                "labels": {kpi_id: labels_by_kpi[kpi_id]},
            },
        )
        for kpi_id in predictions
    }
    pooled_metrics = compute_metrics(predictions, reference_data)
    return per_kpi_metrics, pooled_metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True, type=Path, help="Evolved program defining detect_anomalies")
    parser.add_argument("--data", required=True, type=Path, help="Labeled kpi_heldout.npz archive")
    args = parser.parse_args(argv)

    if not args.program.is_file():
        parser.error(f"candidate program not found: {args.program}")
    if not args.data.is_file():
        parser.error(f"held-out archive not found: {args.data}")

    try:
        detector = load_detector(args.program.resolve())
        per_kpi_metrics, pooled_metrics = score_heldout_data(detector, args.data.resolve())
    except Exception as error:
        parser.exit(2, f"score_heldout.py: error: {error}\n")

    print("Held-out per-KPI scores:")
    for kpi_id, metrics in per_kpi_metrics.items():
        print(
            f"{kpi_id}  F1 {metrics['f1']:.3f}  "
            f"precision {metrics['precision']:.3f}  recall {metrics['recall']:.3f}"
        )
    print(
        f"\nHeld-out pooled F1 {pooled_metrics['f1']:.3f}  "
        f"precision {pooled_metrics['precision']:.3f}  recall {pooled_metrics['recall']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())