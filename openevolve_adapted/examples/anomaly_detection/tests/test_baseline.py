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
            api_base='http://127.0.0.1:8000/v1', max_tokens=4096, workers=1)))

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
        self.assertIn('pointwise F1 pooled over all KPIs',prompt['system'])
        self.assertIn('complete new contents of the EVOLVE-BLOCK',prompt['user'])
        self.assertNotIn('<<<<<<< SEARCH',prompt['user'])

    def test_worker_randomness_repeats_by_seed(self):
        def samples(seed):
            config=self.config(seed)
            controller=ProcessParallelController(config,'unused',None)
            _worker_init(controller._serialize_config(config),'unused')
            return [random.random(),float(np.random.random())]
        self.assertEqual(samples(42),samples(42))
        self.assertNotEqual(samples(42),samples(43))

class RequestSeedTests(unittest.TestCase):
    """Each iteration needs its own sampling seed, and failed edits are dropped."""

    def run_iterations(self, seed, replies):
        import openevolve.process_parallel as pp
        from openevolve.database import Program
        config = BaselineTests().config(seed)
        config.random_seed, config.language = seed, 'python'
        code = (EXAMPLE / 'initial_programs/initial_program.py').read_text()
        seeds, evaluated = [], []

        class FakeLLM:
            async def generate_with_context(self, system_message, messages, **kwargs):
                seeds.append(kwargs.get('seed'))
                return replies[len(seeds) - 1]

        class FakeEvaluator:
            async def evaluate_program(self, child_code, child_id):
                evaluated.append(child_code)
                return {'combined_score': 0.5}

            def get_pending_artifacts(self, child_id):
                return None

        parent = Program(id='parent', code=code, language='python', metrics={'combined_score': 0.5})
        snapshot = {'programs': {'parent': parent.to_dict()}, 'islands': [['parent']],
                    'current_island': 0, 'artifacts': {}}
        saved = {name: getattr(pp, name, None) for name in (
            '_worker_config', '_worker_llm_ensemble', '_worker_prompt_sampler', '_worker_evaluator')}
        pp._worker_config, pp._worker_llm_ensemble = config, FakeLLM()
        pp._worker_prompt_sampler, pp._worker_evaluator = PromptSampler(config.prompt), FakeEvaluator()
        try:
            results = [pp._run_iteration_worker(i + 1, snapshot, 'parent', []) for i in range(len(replies))]
        finally:
            for name, value in saved.items():
                setattr(pp, name, value)
        return seeds, evaluated, results

    def test_same_prompt_gets_a_new_seed_each_iteration(self):
        old = 'threshold = 3 * np.sqrt(np.mean(valid_errors**2))'
        edit = f'<<<<<<< SEARCH\n    {old}\n=======\n    threshold = 4 * np.std(valid_errors)\n>>>>>>> REPLACE'
        seeds, evaluated, results = self.run_iterations(42, [edit, edit, edit])
        self.assertEqual(len(set(seeds)), 3)
        self.assertTrue(all(isinstance(seed, int) and 0 <= seed < 2**31 for seed in seeds))
        self.assertEqual(seeds, self.run_iterations(42, [edit, edit, edit])[0])
        self.assertNotEqual(seeds, self.run_iterations(43, [edit, edit, edit])[0])
        self.assertEqual(len(evaluated), 3)
        self.assertTrue(all(result.error is None for result in results))

    def test_block_rewrite_without_diff_markers_is_graded(self):
        block = ('def helper(x):\n    return np.abs(x)\n\n\n'
                 'def detect_anomalies(error: np.ndarray, missing: np.ndarray) -> np.ndarray:\n'
                 '    return (~missing) & (helper(error) > 1.0)')
        whole = '# EVOLVE-BLOCK-START\n' + block + '\n# EVOLVE-BLOCK-END\n\ndef main():\n    pass'
        replies = [f'My idea.\n```python\n{block}\n```', f'```python\n{whole}\n```', 'Use a rolling median instead.']
        seeds, evaluated, results = self.run_iterations(42, replies)
        self.assertEqual(len(evaluated), 2)
        self.assertEqual(evaluated[0], evaluated[1])
        compile(evaluated[0], 'child', 'exec')
        self.assertIn('def helper', evaluated[0])
        self.assertEqual(evaluated[0].count('def main'), 1)
        self.assertEqual(evaluated[0].count('# EVOLVE-BLOCK-START'), 1)
        self.assertIn('No valid diffs', results[2].error)

    def test_history_shows_only_the_evolve_block(self):
        import openevolve.process_parallel as pp
        captured = {}
        original = PromptSampler.build_prompt
        def spy(sampler, **kwargs):
            captured.update(kwargs)
            return original(sampler, **kwargs)
        PromptSampler.build_prompt = spy
        try:
            self.run_iterations(42, ['no edit'])
        finally:
            PromptSampler.build_prompt = original
        self.assertTrue(captured['top_programs'])
        self.assertNotIn('def main', captured['top_programs'][0]['code'])
        self.assertIn('def detect_anomalies', captured['top_programs'][0]['code'])
        self.assertIn('def main', captured['current_program'])

    def test_unmatched_patch_is_an_error_and_is_not_graded(self):
        unmatched = '<<<<<<< SEARCH\n    not in the program\n=======\n    pass\n>>>>>>> REPLACE'
        seeds, evaluated, results = self.run_iterations(42, [unmatched])
        self.assertEqual(evaluated, [])
        self.assertIn('did not change', results[0].error)


if __name__=='__main__':
    unittest.main()
