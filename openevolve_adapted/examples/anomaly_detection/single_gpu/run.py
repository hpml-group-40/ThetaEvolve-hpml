"""Run the anomaly-detection loop against a local single-GPU vLLM server."""

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

from serve import DEFAULT_MODEL

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
OPENEVOLVE_DIR = EXAMPLE_DIR.parents[1]


def resolved_config(args):
    config = yaml.safe_load((EXAMPLE_DIR / "configs/config_anomaly_detection_vllm.yaml").read_text())
    config["random_seed"] = args.seed
    config["llm"].update(
        models=[{"name": args.model, "weight": 1.0}],
        api_base=args.api_base, api_key="EMPTY", max_tokens=4096, retries=0,
    )
    config["database"]["num_islands"] = 1
    config["evaluator"]["parallel_evaluations"] = 1
    # The resolved YAML lives in outputs/, so resolve staging paths explicitly.
    config["evaluator"]["copy_files"] = [
        {str(EXAMPLE_DIR / "data/kpi_search.npz"): "data/kpi_search.npz"}
    ]
    return config


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}
    with urlopen(Request(url, data=data, headers=headers), timeout=120) as response:
        return json.load(response)


def wait_for_run(command, environment, budget_seconds):
    """Use the existing loop's target stop and the launcher's 20s cleanup policy."""
    process = subprocess.Popen(command, cwd=OPENEVOLVE_DIR, env=environment, start_new_session=(os.name == "posix"))
    reason = "finished"
    try:
        return process.wait(timeout=budget_seconds), reason
    except subprocess.TimeoutExpired:
        reason = "wall_timeout"
        print(f"{budget_seconds}s budget reached; allowing 20s for cleanup.", flush=True)
    except KeyboardInterrupt:
        reason = "interrupted"
    # Signal the controller first so it can save the current best program.
    process.terminate()
    try:
        process.wait(timeout=20)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
    return (124 if reason == "wall_timeout" else 130), reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "profile", "benchmark"), default="smoke")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-seconds", type=int, default=600)
    parser.add_argument("--iterations", type=int, help="Defaults to 2 for smoke/profile; high ceiling for benchmark")
    parser.add_argument("--target-f1", type=float, help="Defaults to 1.0 for smoke/profile and 0.85 for benchmark")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-command", action="store_true", help="Print only; no data, server or GPU needed")
    args = parser.parse_args(argv)
    args.api_base = args.api_base.rstrip("/")
    if args.target_f1 is None:
        args.target_f1 = 0.85 if args.mode == "benchmark" else 1.0
    if args.iterations is None:
        args.iterations = 2147483647 if args.mode == "benchmark" else 2
    if args.budget_seconds < 1 or args.iterations < 1 or not 0 <= args.target_f1 <= 1:
        parser.error("Budget/iterations must be positive; target F1 must lie in [0, 1]")
    if args.mode == "profile":
        from urllib.parse import urlsplit
        base = urlsplit(args.api_base)
        if base.scheme != "http" or base.hostname not in {"127.0.0.1", "localhost", "::1"} or base.path != "/v1":
            parser.error("GPU profiling needs a local vLLM http://localhost:<port>/v1 URL")
    output = args.output or Path("outputs") / f"single_gpu_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}_{args.seed}"
    if not output.is_absolute():
        output = OPENEVOLVE_DIR / output
    output = output.resolve()
    config_path = output / "resolved_config.yaml"
    command = [
        sys.executable, str(OPENEVOLVE_DIR / "openevolve-run.py"),
        str(EXAMPLE_DIR / "initial_programs/initial_program.py"),
        str(EXAMPLE_DIR / "evaluators/evaluator_modular.py"),
        "--config", str(config_path), "--iterations", str(args.iterations),
        "--target-score", str(args.target_f1), "--output", str(output),
    ]
    print(shlex.join(command), flush=True)
    if args.print_command:
        print(yaml.safe_dump(resolved_config(args), sort_keys=False))
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}; choose a new --output")
    for name in ("kpi_search.npz", "kpi_search_labels.npz"):
        if not (EXAMPLE_DIR / "data" / name).is_file():
            parser.error(f"Missing {EXAMPLE_DIR / 'data' / name}")
    models = request_json(args.api_base + "/models")
    if args.model not in {model["id"] for model in models["data"]}:
        parser.error(f"Server is not serving {args.model}; use the same --model in serve.py and run.py")
    # Equal warmup for all modes, outside the evolution timer and GPU trace.
    request_json(args.api_base + "/chat/completions", {
        "model": args.model, "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8, "temperature": 0, "seed": args.seed,
    })
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(resolved_config(args), sort_keys=False))
    environment = os.environ.copy()
    environment["OPENEVOLVE_CONFIG_PATH"] = str(config_path)
    environment["PYTHONPATH"] = str(OPENEVOLVE_DIR) + os.pathsep + environment.get("PYTHONPATH", "")
    # Candidate execution invokes python/python3 through PATH. Keep it in the
    # launcher's environment even when run.py was called by an absolute venv path.
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    for key in ("HPML_TRACE_DIR", "HPML_VLLM_PROFILE"):
        environment.pop(key, None)
    trace_directory = output / "traces"
    if args.mode == "profile":
        environment.update(HPML_TRACE_DIR=str(trace_directory), HPML_VLLM_PROFILE="1")
    metadata = {
        "mode": args.mode, "seed": args.seed, "target_dev_f1": args.target_f1,
        "budget_seconds": args.budget_seconds, "iterations_ceiling": args.iterations,
        "command": command, "warmup_outside_budget": True,
        "data_sha256": {
            name: hashlib.sha256((EXAMPLE_DIR / "data" / name).read_bytes()).hexdigest()
            for name in ("kpi_search.npz", "kpi_search_labels.npz")
        },
    }
    environment["HPML_TARGET_STATUS_PATH"] = str(output / "target_status.json")
    environment["HPML_RUN_START_NS"] = str(time.perf_counter_ns())
    start = time.perf_counter()
    returncode, reason = wait_for_run(command, environment, args.budget_seconds)
    metadata.update(exit_code=returncode, stop_reason=reason,
                    wall_seconds_including_cleanup=time.perf_counter() - start)
    target_path = output / "target_status.json"
    target = json.loads(target_path.read_text()) if target_path.exists() else None
    reached = bool(target and target["elapsed_seconds"] <= args.budget_seconds)
    metadata["target_reached_within_budget"] = reached
    metadata["time_to_target_seconds"] = target["elapsed_seconds"] if reached else None
    if reached:
        metadata["stop_reason"] = "target_reached"
    if args.mode == "profile":
        sys.path.insert(0, str(OPENEVOLVE_DIR))
        from openevolve.utils.trace_utils import _profile_endpoint, merge_traces
        status_path = trace_directory / "gpu_profile_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        # A force-killed client thread cannot execute its finally block.
        if (trace_directory / "gpu-profile.claim").exists() and not status.get("stopped"):
            try:
                _profile_endpoint(args.api_base, "stop_profile", "EMPTY")
            except Exception as error:
                print(f"Profiler cleanup: {error}", file=sys.stderr)
        events = merge_traces(trace_directory, output / "loop_trace.json")
        metadata["trace_events"] = len(events)
        metadata["gpu_profile"] = status or {"status": "not_started"}
        if not events:
            print("CPU trace is empty; check trace output permissions.", file=sys.stderr)
            returncode = returncode or 1
        if status.get("status") != "complete":
            print("GPU profile did not complete. Check server profiler settings and whether an LLM call ran.", file=sys.stderr)
            returncode = returncode or 1
    metadata["exit_code"] = returncode
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Outputs: {output}")
    if args.mode == "profile":
        print("Diagnostic profile: exclude this run from reported speed comparisons.")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
