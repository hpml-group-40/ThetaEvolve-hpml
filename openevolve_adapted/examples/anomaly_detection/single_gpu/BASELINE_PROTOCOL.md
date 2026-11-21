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

On the configured VM, start the server in one terminal:

    cd /home/hq2230/ThetaEvolve-hpml
    source .venv-vllm/bin/activate
    python /home/hq2230/vm-bootstrap/serve-thetaevolve.py

When the server is ready, run the suite in a second terminal:

    cd /home/hq2230/ThetaEvolve-hpml
    bash openevolve_adapted/examples/anomaly_detection/single_gpu/run_baseline_suite.sh

Per-seed logs, resolved configuration, data hashes, best detector, software versions,
server command, source diff, and summary are saved under the suite output directory.
A raw runner exit code of 124 means the fixed budget was reached; the wrapper
requires the run metadata and best detector to exist before treating it as success.

Prompt compatibility fix: use the task instruction as the actual system message and require one literal SEARCH/REPLACE patch of detect_anomalies. The same frozen prompt applies to every measured configuration.

Initial source indentation is normalized from tabs to four spaces, with identical Python AST. This avoids silent unmatched patches when Qwen copies indentation as spaces; detector logic is unchanged.
