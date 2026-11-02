"""Validation and search-split scoring for anomaly-detection candidates."""

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_reference_data(data_dir: Optional[Path] = None) -> Dict[str, Dict[str, np.ndarray]]:
	"""Load search labels and masks independently of candidate output."""
	data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
	search_path = data_dir / "kpi_search.npz"
	labels_path = data_dir / "kpi_search_labels.npz"

	with np.load(search_path, allow_pickle=False) as search, np.load(
		labels_path, allow_pickle=False
	) as label_archive:
		kpi_ids = sorted(key[: -len("_value")] for key in search.files if key.endswith("_value"))
		label_ids = sorted(
			key[: -len("_label")] for key in label_archive.files if key.endswith("_label")
		)
		if not kpi_ids or kpi_ids != label_ids:
			raise ValueError("Search KPI IDs and label KPI IDs must match and be non-empty")

		missing_by_kpi = {}
		labels_by_kpi = {}
		for kpi_id in kpi_ids:
			missing_key = f"{kpi_id}_missing"
			label_key = f"{kpi_id}_label"
			if missing_key not in search.files or label_key not in label_archive.files:
				raise ValueError(f"Missing search mask or labels for KPI {kpi_id}")

			missing = search[missing_key]
			labels = label_archive[label_key]
			if missing.ndim != 1 or labels.ndim != 1 or len(missing) != len(labels):
				raise ValueError(f"Search mask and labels must be aligned 1-D arrays for KPI {kpi_id}")
			if missing.dtype.kind != "b":
				raise ValueError(f"Missing mask must be boolean for KPI {kpi_id}")
			if not np.isin(labels, (0, 1)).all():
				raise ValueError(f"Labels must contain only 0 or 1 for KPI {kpi_id}")

			missing_by_kpi[kpi_id] = missing
			labels_by_kpi[kpi_id] = labels.astype(bool, copy=False)

	return {"missing": missing_by_kpi, "labels": labels_by_kpi}


def validate_solution(
	solution_data: object,
	reference_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Tuple[bool, str]:
	"""Validate a mapping of KPI IDs to full-length binary prediction arrays."""
	try:
		reference_data = reference_data or load_reference_data()
		if not isinstance(solution_data, Mapping):
			return False, "Solution must map KPI IDs to prediction arrays"

		expected_ids = set(reference_data["labels"])
		actual_ids = set(solution_data)
		if actual_ids != expected_ids:
			return False, "Prediction KPI IDs do not match search data"

		for kpi_id in sorted(expected_ids):
			predictions = np.asarray(solution_data[kpi_id])
			expected_length = len(reference_data["labels"][kpi_id])
			if predictions.ndim != 1 or len(predictions) != expected_length:
				return False, f"Predictions must be a length-{expected_length} array for KPI {kpi_id}"
			if predictions.dtype.kind not in "biuf":
				return False, f"Predictions must contain binary numeric values for KPI {kpi_id}"
			if predictions.dtype.kind == "f" and not np.isfinite(predictions).all():
				return False, f"Predictions contain non-finite values for KPI {kpi_id}"
			if not np.isin(predictions, (0, 1)).all():
				return False, f"Predictions must contain only 0 or 1 for KPI {kpi_id}"

		return True, "Valid anomaly predictions"
	except Exception as error:
		return False, f"Validation error: {error}"


def compute_metrics(
	solution_data: object,
	reference_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, float]:
	"""Recompute pooled pointwise F1, precision, and recall on non-missing points."""
	reference_data = reference_data or load_reference_data()
	is_valid, error_message = validate_solution(solution_data, reference_data)
	if not is_valid:
		raise ValueError(error_message)

	true_positives = false_positives = false_negatives = 0
	for kpi_id, raw_predictions in solution_data.items():
		keep = ~reference_data["missing"][kpi_id]
		predictions = np.asarray(raw_predictions, dtype=bool)[keep]
		labels = reference_data["labels"][kpi_id][keep]
		true_positives += int((predictions & labels).sum())
		false_positives += int((predictions & ~labels).sum())
		false_negatives += int((~predictions & labels).sum())

	precision = true_positives / max(true_positives + false_positives, 1)
	recall = true_positives / max(true_positives + false_negatives, 1)
	f1 = 2 * precision * recall / max(precision + recall, 1e-12)
	return {"f1": f1, "precision": precision, "recall": recall}
