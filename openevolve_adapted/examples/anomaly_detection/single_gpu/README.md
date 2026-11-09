# Single GPU setup and traces

This uses the OpenAI-compatible evolution loop directly: one vLLM server generates edits on one NVIDIA GPU, and the existing Python evaluator scores them on the CPU. It does not launch the original SLIME/SGLang/Ray training stack.

The starter model is `Qwen/Qwen2.5-Coder-1.5B-Instruct`. It is a small coding model for checking the setup, not a claim that it can reach the team's target F1. Use the same `--model` in both launchers if the team selects another model.

## Tests locally

These checks are quick, need no dataset or GPU, and do not contact an LLM:

```bash
python -m unittest discover -s openevolve_adapted/examples/anomaly_detection/tests -p test_profiling.py -v
python openevolve_adapted/examples/anomaly_detection/single_gpu/serve.py --print-command
python openevolve_adapted/examples/anomaly_detection/single_gpu/run.py --mode profile --print-command
```

The tests use a local mock HTTP server. Passing them verifies trace files and profiler start/stop behavior; it does not verify CUDA inference.

## GPU VM and environment

Before installing Python packages, `nvidia-smi` must work. The driver must support the CUDA build used by the pinned vLLM wheel; vLLM's installation guide lists its requirements. The scripts and tests can be prepared before provisioning any paid VM.

On that VM, from the repository root, in a fresh environment:

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r openevolve_adapted/examples/anomaly_detection/single_gpu/requirements-vllm.txt
python -m pip install -e ./openevolve_adapted
python -m pip check
```

vLLM, Transformers and the OpenAI SDK are pinned to make the version-specific flags explicit. This is a reproducible starting version, not a request to use the latest release. Let vLLM install its matching PyTorch/CUDA dependencies. Do not install the root repository's training requirements for this path.

Put the real `kpi_search.npz` and `kpi_search_labels.npz` in `openevolve_adapted/examples/anomaly_detection/data/`. Keep the held-out archive for the separate final scoring command. Data and outputs are already ignored.

## Diagnostic profile: two terminals on the same VM

The following commands are the later GPU check. The first launch downloads and loads the model; it may take several minutes. Both terminals start from `openevolve_adapted/` with `.venv-vllm` activated.

Terminal 1:

```bash
python examples/anomaly_detection/single_gpu/serve.py \
  --gpu 0 --profile-dir outputs/gpu_trace_smoke
```

Once the server says it is ready, terminal 2:

```bash
python examples/anomaly_detection/single_gpu/run.py \
  --mode profile --iterations 2 --budget-seconds 600 \
  --seed 42 --output outputs/trace_smoke
```

The client checks the served model and warms it up outside the evolution timer. It then evaluates the initial program, asks for up to two edits, and records:

| File | What it contains |
| --- | --- |
| `outputs/trace_smoke/loop_trace.json` | CPU spans from the controller and worker processes |
| `outputs/gpu_trace_smoke/*.pt.trace.json.gz` | vLLM's CPU/CUDA trace of the first generation call |
| `outputs/gpu_trace_smoke/server_launch.json` | Server command, profiler settings, software version and GPU information |
| `outputs/trace_smoke/resolved_config.yaml` | Exact evolution settings used in this run |
| `outputs/trace_smoke/run_metadata.json` | Seed, data hashes, termination state and profile completion state |

Open `loop_trace.json` and the vLLM GPU trace separately in [Perfetto](https://ui.perfetto.dev/). The client trace shows parent selection, database snapshots, prompt building, LLM waiting time, diff application, candidate evaluation, database updates and checkpoints. CUDA kernels appear in the vLLM trace. Do not add nested span durations together or interpret all client waiting time as GPU computation. No prompt text or API keys are added to trace events.

The profiler is enabled only for the first generation call across workers and is stopped in a `finally` block. It can therefore produce a large trace for a 4096-token completion, but does not record an unlimited sequence of generations. The launcher attempts to stop it after a hard client timeout too. If the initial program already reaches the target, no LLM request runs and the launcher reports that a GPU profile was not produced. Use an empty output directory for each run.

Profiling changes latency. **Exclude diagnostic runs from speed comparisons.** If the model returns an invalid diff, the trace can still be valid; verify that the logs also show a successfully scored child before declaring evolution ready.

## Timing runs after the profile works

Stop the profiling server with Ctrl+C and restart it without `--profile-dir`:

```bash
python examples/anomaly_detection/single_gpu/serve.py --gpu 0
```

Then run one seeded trial (target and budget below are examples):

```bash
python examples/anomaly_detection/single_gpu/run.py \
  --mode benchmark --target-f1 0.85 --budget-seconds 3600 \
  --seed 42 --output outputs/reference_seed42
```

`benchmark` disables both the client trace and GPU profiler controls. The launcher records `time_to_target_seconds` when the controller confirms the target, using the same host's monotonic clock. The clock starts just before the evolution process launches, including client startup and the initial evaluation; model download, server startup, the fixed warmup, cleanup and held-out scoring are excluded. A target observed after the budget is a miss and has a null time to target. `wall_seconds_including_cleanup` is a separate diagnostic value.

A deadline gets the same 20-second cleanup allowance as Sonya's launcher, followed by process-group termination if needed. Timeout exit status is 124 even if cleanup exits normally. An iteration ceiling is only a safety limit. The team should keep the model/revision, prompts, data, target, budget, sampling settings and hardware fixed across comparisons, and use the same 2-3 seeds for each setup. Seeds reduce variation; they do not guarantee identical generated code. Use `--revision <model-commit>` in the server launcher to fix the model revision.

The existing held-out scorer remains a separate final step for each completed trial's selected best program. From `openevolve_adapted/`:

```bash
python examples/anomaly_detection/score_heldout.py \
  --program outputs/reference_seed42/best/best_program.py \
  --data /absolute/path/to/kpi_heldout.npz
```

## What is on or off

These are explicit **starter settings**, not an assertion that every vLLM optimization is disabled. BF16 weights/KV and FlashAttention are already active. Before reporting a strict full-precision baseline, the team must choose a supported FP32 attention/model configuration and validate it separately, or define these features as fixed infrastructure in the baseline. This launcher does not provide or validate an FP32/ordinary-attention comparison.

| Feature | Starter setting | Control for later experiments |
| --- | --- | --- |
| Concurrent request batching | One sequence at a time | `--max-num-seqs 1`; increase alongside client workers to expose batching |
| Parallel evolution/evaluation | One worker | `evaluator.parallel_evaluations: 1`; in this repo a worker performs generation plus evaluation, so changing it changes both |
| Islands | One | `database.num_islands`; multiple islands do not imply multiple GPUs |
| Within-request KV cache | On | Fundamental to the selected vLLM generation path; there is no supported ordinary off switch here |
| Cross-request prefix cache | Off | `--no-enable-prefix-caching`; replace with `--enable-prefix-caching` |
| CUDA graphs | Off | `--enforce-eager`; enabling requires revisiting compilation/graph settings together |
| `torch.compile` | Off | `--compilation-config '{"level": 0}'`; V1 otherwise normally uses optimized compilation |
| Attention backend | FlashAttention | `VLLM_ATTENTION_BACKEND=FLASH_ATTN`; alternative backends require model/dtype/GPU compatibility checks |
| Chunked prefill | Off | `--no-enable-chunked-prefill`; replace with `--enable-chunked-prefill` |
| Weight precision | BF16, unquantized | Explicit `--dtype bfloat16`; an FP32 or quantized experiment needs a separately validated server setup |
| KV precision | BF16 | `--kv-cache-dtype auto` follows this model dtype; FP8 is a separate optimization |
| KV sizing | 80% GPU-memory allowance; 8192 total tokens | `--gpu-memory-utilization 0.80`, `--max-model-len 8192`; allowance includes model/runtime and is not all KV memory |
| Speculative decoding | Off | No `--speculative-config` is passed |
| Output/sampling | 4096 tokens, temperature 0.7, top-p 0.95 | Resolved YAML; these are fixed across comparisons |
| Evolution | Diff edits, population 1000 | Inherited from Sonya's configuration |
| LLM retries | No outer loop retries | Launcher sets `llm.retries: 0`; the OpenAI SDK still has its own default transport retries |

The 8192-token context includes prompt **and** output. Leave room for the 4096 output-token cap; if evolved code/prompts grow too large, choose a larger context and matching batched-token limit before fixing the experiment configuration.

Other possible experiments after identifying the bottleneck are CPU verifier data reuse, reducing repeated database snapshots, and avoiding excessive checkpoint/plotting work. Measure these first. Changing the model, token cap, population, prompts or acceptance logic also changes the search problem and should not be presented as a pure runtime optimization.

## References

- [vLLM 0.11.0 GPU installation](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html)
- [vLLM 0.11.0 profiling](https://docs.vllm.ai/en/v0.11.0/contributing/profiling.html)
- [vLLM 0.11.0 server arguments](https://docs.vllm.ai/en/v0.11.0/cli/serve.html)
- [Qwen model card and serving example](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [GCP GPU driver installation](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
