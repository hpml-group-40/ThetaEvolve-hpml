"""Regression checks for baseline prompt and experiment seed propagation."""
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE / 'single_gpu'))
import run as runner
from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler
from openevolve.process_parallel import ProcessParallelController, _worker_init

class BaselineTests(unittest.TestCase):
    def config(self, seed):
        return Config.from_dict(runner.resolved_config(SimpleNamespace(
            seed=seed, model='Qwen/Qwen2.5-Coder-7B-Instruct',
            api_base='http://127.0.0.1:8000/v1')))

    def test_each_seed_reaches_database(self):
        for seed in (42,43,44):
            config=self.config(seed)
            self.assertEqual(config.random_seed,seed)
            self.assertEqual(config.database.random_seed,seed)

    def test_task_instruction_reaches_actual_prompt(self):
        config=self.config(42)
        prompt=PromptSampler(config.prompt).build_prompt(
            current_program=(EXAMPLE/'initial_programs/initial_program.py').read_text(),
            program_metrics={'combined_score':0.5412038233014724})
        self.assertIn('Optimize pooled pointwise F1',prompt['system'])
        self.assertIn('<<<<<<< SEARCH',prompt['user'])

    def test_worker_randomness_repeats_by_seed(self):
        def samples(seed):
            config=self.config(seed)
            controller=ProcessParallelController(config,'unused',None)
            _worker_init(controller._serialize_config(config),'unused')
            return [random.random(),float(np.random.random())]
        self.assertEqual(samples(42),samples(42))
        self.assertNotEqual(samples(42),samples(43))

if __name__=='__main__':
    unittest.main()
