import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openevolve.config import load_config
from openevolve.evaluator import Evaluator
from examples.anomaly_detection.initial_programs.initial_program import resolve_search_path


class EvaluationFileStagingTests(unittest.TestCase):
    def make_evaluator(self, evaluator_dir: Path, copy_files):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.config = SimpleNamespace(copy_folders=[], copy_files=copy_files)
        evaluator.evaluation_file = str(evaluator_dir / "evaluator.py")
        return evaluator

    def test_mapping_stages_only_the_requested_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evaluator_dir = root / "example" / "evaluators"
            data_dir = root / "example" / "data"
            workspace = root / "workspace"
            evaluator_dir.mkdir(parents=True)
            data_dir.mkdir()
            workspace.mkdir()
            (evaluator_dir / "evaluator.py").touch()
            (data_dir / "kpi_search.npz").write_bytes(b"features")
            (data_dir / "kpi_search_labels.npz").write_bytes(b"labels")
            evaluator = self.make_evaluator(
                evaluator_dir,
                [{"../data/kpi_search.npz": "data/kpi_search.npz"}],
            )

            Evaluator._copy_evaluation_files(evaluator, str(workspace))

            self.assertEqual((workspace / "data" / "kpi_search.npz").read_bytes(), b"features")
            self.assertFalse((workspace / "data" / "kpi_search_labels.npz").exists())

    def test_legacy_string_keeps_same_relative_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evaluator_dir = root / "evaluators"
            workspace = root / "workspace"
            (evaluator_dir / "resources").mkdir(parents=True)
            workspace.mkdir()
            (evaluator_dir / "evaluator.py").touch()
            (evaluator_dir / "resources" / "reference.json").write_text("{}")
            evaluator = self.make_evaluator(evaluator_dir, ["resources/reference.json"])

            Evaluator._copy_evaluation_files(evaluator, str(workspace))

            self.assertEqual(
                (workspace / "resources" / "reference.json").read_text(), "{}"
            )

    def test_mapping_cannot_write_outside_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evaluator_dir = root / "evaluators"
            source_dir = root / "data"
            workspace = root / "workspace"
            evaluator_dir.mkdir()
            source_dir.mkdir()
            workspace.mkdir()
            (evaluator_dir / "evaluator.py").touch()
            (source_dir / "features.npz").write_bytes(b"features")
            evaluator = self.make_evaluator(
                evaluator_dir, [{"../data/features.npz": "../outside.npz"}]
            )

            Evaluator._copy_evaluation_files(evaluator, str(workspace))

            self.assertFalse((root / "outside.npz").exists())


class CandidateDataResolutionTests(unittest.TestCase):
    def test_prefers_search_features_staged_beside_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            data_dir = workspace / "data"
            data_dir.mkdir()
            program_file = workspace / "program.py"
            program_file.touch()
            archive = data_dir / "kpi_search.npz"
            archive.write_bytes(b"staged features")

            self.assertEqual(resolve_search_path(program_file), archive.resolve())

    def test_falls_back_to_checked_in_example_data(self):
        archive = resolve_search_path()

        self.assertTrue(archive.is_file())
        self.assertEqual(archive.name, "kpi_search.npz")

    def test_initial_program_scores_through_temporary_evaluation_workspace(self):
        openevolve_root = Path(__file__).resolve().parents[3]
        example_dir = openevolve_root / "examples" / "anomaly_detection"
        config_path = example_dir / "configs" / "config_anomaly_detection.yaml"
        evaluator_path = example_dir / "evaluators" / "evaluator_modular.py"
        program_path = example_dir / "initial_programs" / "initial_program.py"
        env = {
            "OPENEVOLVE_CONFIG_PATH": str(config_path),
            "PYTHONPATH": str(openevolve_root),
            "PATH": str(Path(sys.executable).parent)
            + os.pathsep
            + os.environ.get("PATH", ""),
        }

        with patch.dict(os.environ, env):
            config = load_config(config_path)
            config.evaluator.collect_runtime_environments = False
            evaluator = Evaluator(config.evaluator, str(evaluator_path))
            metrics = asyncio.run(
                evaluator.evaluate_program(
                    program_path.read_text(), program_id="anomaly-data-staging-test"
                )
            )

        self.assertEqual(metrics.get("validity"), 1.0)
        self.assertAlmostEqual(metrics["combined_score"], 0.5412038233014724)


if __name__ == "__main__":
    unittest.main()