import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openevolve.controller import OpenEvolve
from openevolve.process_parallel import has_reached_target_score


class TargetScoreTests(unittest.TestCase):
    def test_target_uses_combined_score_not_auxiliary_metric_average(self):
        metrics = {
            "combined_score": 0.82,
            "f1": 0.82,
            "precision": 1.0,
            "recall": 0.70,
            "validity": 1.0,
            "eval_time": 0.1,
        }

        self.assertTrue(has_reached_target_score(metrics, 0.8))
        self.assertFalse(has_reached_target_score(metrics, 0.83))

    def test_legacy_metrics_fall_back_to_numeric_average(self):
        self.assertTrue(has_reached_target_score({"score": 0.8}, 0.75))
        self.assertFalse(has_reached_target_score({"message": "invalid"}, 0.1))

    def test_initial_program_target_skips_generations(self):
        class FakeDatabase:
            def __init__(self):
                self.programs = {}
                self.best_program_id = None
                self.last_iteration = 0

            def add(self, program):
                self.programs[program.id] = program
                self.best_program_id = program.id

            def get(self, program_id):
                return self.programs[program_id]

            def get_best_program(self, metric=None):
                return max(
                    self.programs.values(),
                    key=lambda program: program.metrics[metric or "combined_score"],
                    default=None,
                )

            def log_island_status(self):
                pass

        class FakeEvaluator:
            async def evaluate_program(self, code, program_id):
                return {"combined_score": 0.85, "precision": 0.9, "recall": 0.8}

            def cleanup_runtime_environments(self):
                pass

        class FakeParallelController:
            def __init__(self, config, evaluation_file, database):
                self.shutdown_event = threading.Event()
                self.max_iterations = None

            def start(self):
                pass

            async def run_evolution(
                self, start_iteration, max_iterations, target_score, checkpoint_callback=None
            ):
                assert max_iterations == 0
                self.max_iterations = max_iterations

            def stop(self):
                pass

        evolve = OpenEvolve.__new__(OpenEvolve)
        evolve.config = SimpleNamespace(
            max_iterations=100,
            random_seed=None,
            language="python",
            database=SimpleNamespace(num_islands=1),
            checkpoint_interval=10,
        )
        evolve.initial_program_path = "initial_program.py"
        evolve.evaluation_file = "evaluator.py"
        evolve.initial_program_code = "pass"
        evolve.file_extension = ".py"
        evolve.database = FakeDatabase()
        evolve.evaluator = FakeEvaluator()
        evolve.parallel_controller = None
        evolve.iteration_programs = {}
        evolve.performance_tracker = SimpleNamespace(finalize=lambda: None)
        with patch("openevolve.controller.ProcessParallelController", FakeParallelController):
            with patch.object(evolve, "_save_best_program"):
                with patch.object(evolve, "_create_performance_visualizations"):
                    result = asyncio.run(evolve.run(iterations=100, target_score=0.8))

        self.assertAlmostEqual(result.metrics["combined_score"], 0.85)


if __name__ == "__main__":
    unittest.main()