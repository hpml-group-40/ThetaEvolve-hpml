"""One-shot modular evaluator for search-split anomaly detection."""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from openevolve.modular_utils.evaluation_controller import (
	BaseProblemEvaluator,
	create_evaluation_controller,
	get_current_problem_config,
)

if __package__:
	from .verify import compute_metrics, load_reference_data, validate_solution
else:
	from verify import compute_metrics, load_reference_data, validate_solution


class AnomalyDetectionProblemEvaluator(BaseProblemEvaluator):
	"""Score candidate flags against trusted search labels and masks."""

	def __init__(self, reference_data: Dict[str, Dict[str, Any]]) -> None:
		self.reference_data = reference_data

	def validate_solution(self, solution_data: Any) -> Tuple[bool, str]:
		return validate_solution(solution_data, self.reference_data)

	def compute_objective_score(self, full_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
		current_solution = full_data.get("current_solution") or {}
		solution_data = current_solution.get("data")
		if solution_data is None:
			raise ValueError("No candidate predictions found")

		is_valid, error_message = self.validate_solution(solution_data)
		if not is_valid:
			raise ValueError(error_message)

		metrics = compute_metrics(solution_data, self.reference_data)
		return metrics["f1"], metrics


def evaluate(program_path: str, temp_dir: str = None):
	"""Run one candidate and return its verifier-backed evaluation result."""
	problem_evaluator = AnomalyDetectionProblemEvaluator(load_reference_data())
	evaluation_controller = create_evaluation_controller(
		problem_evaluator, get_current_problem_config()
	)
	return evaluation_controller.evaluate(program_path, temp_dir)


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python evaluator_modular.py <program_path>")
		raise SystemExit(2)

	result = evaluate(str(Path(sys.argv[1]).resolve()))
	metrics = result.metrics
	score = metrics.get("combined_score")
	print(f"Pooled F1: {score}")
	print(f"Precision: {metrics.get('precision')}")
	print(f"Recall: {metrics.get('recall')}")
	print(f"Validity: {metrics.get('validity')}")
	if "error" in result.artifacts:
		print(f"Verifier: invalid ({result.artifacts['error']})")
	else:
		print("Verifier: valid")
