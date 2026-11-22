# Group 40 baseline protocol

Frozen before the measured runs on 2026-11-21 UTC.

- Three independent runs: seeds 42, 43, 44; each starts from initial_program.py.
- Search budget: 900 seconds per seed. Model startup and fixed warmup excluded;
  controller startup and initial detector evaluation included. Cleanup gets up to
  20 additional seconds; in-flight results are not accepted after shutdown.
- Primary metric: best accepted search-set pooled pointwise F1 at the deadline.
  Report each seed, mean, and sample standard deviation. Target F1 is 1.0 so an
  ordinary improvement does not stop a run early (perfect F1 may stop early).
- Compare every optimization with the same model/revision, data hashes, seed list,
  prompt, sampling, search budget, initial detector and hardware. Change only the
  named optimization. Do not select new seeds after seeing their scores.
- Qwen/Qwen2.5-Coder-7B-Instruct revision
  c03e6d358207e414f1eca0bb1891e29f1db0e242; one NVIDIA L4.
- vLLM 0.11.0, BF16 weights and KV cache, FlashAttention, context 8192,
  max output 4096, temperature 0.7, top-p 0.95, GPU allowance 0.90.
- Baseline: one worker, one island, max-num-seqs 1; prefix cache off;
  eager execution; compilation off; profiling off. Server seed fixed at 42;
  OpenEvolve derives the LLM request seed from each experiment seed.
- vLLM V1 forces chunked prefill ON despite the launcher's disable flag.
  BF16, FlashAttention, and within-request KV caching are existing infrastructure;
  this is not an unoptimized FP32/no-cache baseline.
- Smoke/timeout checks are excluded from the three measured runs.
- Only search features are staged to candidates. Labels are verifier inputs.
  This runner is not a filesystem security sandbox. Held-out data is not used.
- Before measured runs: enable the task instruction; propagate the seed to the
  database and worker RNGs; save the best accepted detector before blocking on
  worker cleanup at the deadline. Apply these fixes to all future comparisons.

From the repository root, one command starts the pinned server, runs the three
measured seeds, then restarts the server with the kernel profiler for one short
diagnostic run that is excluded from timing:

    bash openevolve_adapted/examples/anomaly_detection/single_gpu/run_baseline_suite.sh

`HPML_BUDGET` (seconds per seed), `HPML_SEEDS` and `HPML_SKIP_PROFILE=1` override
the defaults. Server startup time is recorded in `server.startup.json`.

Per-seed logs, resolved configuration, data hashes, best detector, software versions,
server command, source diff, and summary are saved under the suite output directory.
A raw runner exit code of 124 means the fixed budget was reached; the wrapper
requires the run metadata and best detector to exist before treating it as success.

Prompt compatibility fix: use the task instruction as the actual system message and require one literal SEARCH/REPLACE patch of detect_anomalies. The same frozen prompt applies to every measured configuration.

Initial source indentation is normalized from tabs to four spaces, with identical Python AST. This avoids silent unmatched patches when Qwen copies indentation as spaces; detector logic is unchanged.

## Revision 2026-11-22 (before the re-run; applies to every later comparison)

The 2026-11-21 suite is superseded. Its checkpoints show every repeated prompt
received a byte-identical reply (seed 42: 20 children, 3 distinct programs),
because the loop sent one fixed sampling seed with every request.

- Sampling seed: each LLM request uses a seed derived from (experiment seed,
  iteration). Runs stay reproducible; a re-sampled parent now gets a new reply.
- Failed edits: a patch whose SEARCH text matches nothing used to be graded and
  stored as a copy of its parent. It is now logged and dropped before grading.
- Measured runs record loop spans and sample nvidia-smi (5 Hz) and vLLM /metrics
  (1 Hz). Outputs per seed: `evolution/loop_trace.json` (Perfetto),
  `evolution/trace_summary.json`, `evolution/traces/gpu_samples.csv` and
  `vllm_metrics.csv`. This overhead is identical in every configuration. The
  server-side CUDA kernel profiler stays off in measured runs.
- Diagnostic run: `diagnostic_profile/gpu_kernel_trace/*.pt.trace.json.gz` holds
  CUDA kernels for the first 5 s of one generation call (prefill plus the first
  decode steps); a whole eager reply is too large to open in Perfetto.
- GPU memory allowance 0.90 and `--max-num-seqs 1` are now `serve.py` flags and
  defaults; no files outside the repository are needed.
- Prompt (frozen from this revision): the system message describes the task,
  inputs, scoring and the rules that keep code runnable, and leaves the detection
  method open. The task template in `prompts/` asks for the complete new
  EVOLVE-BLOCK in one code block with no explanation; the generic SEARCH/REPLACE
  examples are gone. The loop accepts either a block rewrite or SEARCH/REPLACE.
- Search history: the two best programs so far are shown (EVOLVE-BLOCK only)
  ahead of the parent program.
- Primary metric: wall-clock time to reach the target search F1. The target is
  the median of the three baseline seeds' best search F1 (`baseline_summary.json`).
  Every run, optimized ones included, uses the full budget with stop target 1.0;
  `best_f1_curve.csv` records each improvement, and time to target is read from
  it. A run that never reaches the target is reported as a miss at the budget.
- Each run's `trace_summary.json` also reports: programs requested, discarded,
  graded, valid and distinct; time to first token, inter-token latency and
  queue/prefill/decode time from vLLM's histograms; decode speed against the
  memory-bandwidth limit; and a minute-by-minute timeline.
  `diagnostic_profile/kernel_summary.json` gives the GPU busy fraction and the
  kernel mix from the kernel trace.
