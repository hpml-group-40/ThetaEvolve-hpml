# Anomaly Detection

For a one-NVIDIA-GPU vLLM setup, optional Perfetto CPU/GPU traces, and commands
for local checks and later GPU runs, see [single_gpu/README.md](single_gpu/README.md).

This example evolves a detector against pooled pointwise F1 on the search split. The verifier applies the missing-point masks and loads labels independently. Held-out data is not used during evolution.

## One-Shot Evaluator Check

From `openevolve_adapted/`, run the initial candidate once without starting evolution or calling an LLM:

```bash
bash examples/anomaly_detection/run_init_example.sh
```

The candidate reads `kpi_search.npz` and writes its prediction payload into the evaluator's temporary workspace. The evaluator reports F1, precision, recall, and validity.

## Evolution Smoke Run

Install the project in the active environment (`pip install -e .`) and select a provider profile. The launcher requires a provider, a target pooled F1 between 0 and 1, and a maximum wall-clock runtime in seconds. For example, `0.85` and `3600` request a stop at pooled F1 0.85 or after one hour, whichever comes first. The target is checked against `combined_score` (pooled pointwise F1 on the search/dev split), not averaged with precision, recall, or timing metrics. The initial candidate is checked too, so no LLM generation starts if it already meets the target.

OpenRouter requires a key in the environment. No key is stored in the YAML:

```bash
export OPENAI_API_KEY="your-key"
bash examples/anomaly_detection/run_evolution.sh openrouter 0.85 3600
```

For vLLM, edit `configs/config_anomaly_detection_vllm.yaml` with the server's reachable OpenAI-compatible `/v1` URL and served model name, then run:

```bash
bash examples/anomaly_detection/run_evolution.sh vllm 0.85 3600
```

Both profiles use the shared OpenAI SDK chat-completions client. The adapter sends a `seed` parameter when configured; set the profile's `random_seed: null` if the server rejects it. A vLLM server with authentication enabled also needs a matching API key configuration.

At the time limit, the launcher requests graceful shutdown and allows up to 20 seconds for cleanup. If the process is still running, it terminates the process group and exits with status 124. OpenEvolve's configured iteration maximum is only a high safety ceiling; target F1 or wall-clock timeout is intended to stop the run first.

Outputs and checkpoints are written under `openevolve_adapted/outputs/`, which is ignored by Git. An alternate output directory can be passed as the third launcher argument.

## Final Held-Out Score

After selecting a detector, score it once against the labeled held-out archive. From `openevolve_adapted/`, pass the chosen program and archive paths explicitly:

```bash
python examples/anomaly_detection/score_heldout.py \
	--program outputs/<run>/best/best_program.py \
	--data ../../time_series_data/kpi_heldout.npz
```

The script calls the candidate's `detect_anomalies(error, missing)` function, excludes missing points, and prints per-KPI and pooled F1, precision, and recall. It reads labels from the held-out archive and does not change or use evolution scores. Run this only for the selected candidate, not during tuning.

## Data Boundary

Only `kpi_search.npz` is staged into each candidate workspace. Search labels remain in the evaluator, and `kpi_heldout.npz` is reserved for a separate final assessment. Candidate execution is not a security sandbox, so do not treat filesystem access as an enforcement boundary.