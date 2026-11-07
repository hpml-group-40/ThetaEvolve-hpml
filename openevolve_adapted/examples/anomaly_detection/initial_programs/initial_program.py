"""Baseline candidate for anomaly detection."""

from pathlib import Path
from typing import Optional

import numpy as np


# EVOLVE-BLOCK-START
def detect_anomalies(error: np.ndarray, missing: np.ndarray) -> np.ndarray:
	"""Flag errors beyond three times the non-missing RMS error."""
	valid_errors = error[~missing]
	if valid_errors.size == 0:
		raise ValueError("KPI has no non-missing errors")

	threshold = 3 * np.sqrt(np.mean(valid_errors**2))
	return (~missing) & (np.abs(error) > threshold)


# EVOLVE-BLOCK-END


def resolve_search_path(program_file: Optional[Path] = None) -> Path:
	"""Find search features in the candidate workspace or source example."""
	program_dir = Path(program_file or __file__).resolve().parent
	candidates = (
		program_dir / "data" / "kpi_search.npz",
		program_dir.parent / "data" / "kpi_search.npz",
	)
	for candidate in candidates:
		if candidate.is_file():
			return candidate
	raise FileNotFoundError(
		"Could not find kpi_search.npz in the candidate workspace or source example: "
		+ ", ".join(str(candidate) for candidate in candidates)
	)


def main() -> None:
	search_path = resolve_search_path()
	predictions = {}

	with np.load(search_path, allow_pickle=False) as search:
		kpi_ids = sorted(key[: -len("_value")] for key in search.files if key.endswith("_value"))
		if not kpi_ids:
			raise ValueError(f"No KPI value arrays found in {search_path}")

		for kpi_id in kpi_ids:
			error = search[f"{kpi_id}_error"]
			missing = search[f"{kpi_id}_missing"]
			if error.ndim != 1 or missing.shape != error.shape or missing.dtype.kind != "b":
				raise ValueError(f"Invalid error or missing arrays for KPI {kpi_id}")
			predictions[kpi_id] = detect_anomalies(error, missing).tolist()

	from openevolve.modular_utils.file_io_controller import save_search_results

	result_path = save_search_results(
		best_perfect_solution=None,
		current_solution={"data": predictions},
		n_kpis=len(predictions),
	)
	if not result_path:
		raise RuntimeError("Failed to save candidate predictions")
	print(f"Saved predictions for {len(predictions)} KPIs")


if __name__ == "__main__":
	main()
