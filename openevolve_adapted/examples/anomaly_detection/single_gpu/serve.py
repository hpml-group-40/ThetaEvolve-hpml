"""Serve a small coding LLM on exactly one NVIDIA GPU with explicit settings."""

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
VLLM_VERSION = "0.11.0"


def build_launch(args):
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--served-model-name", args.model,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--tensor-parallel-size", "1",
        "--pipeline-parallel-size", "1",
        "--data-parallel-size", "1",
        "--dtype", "bfloat16",
        "--kv-cache-dtype", "auto",
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_model_len),
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.80",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "--compilation-config", '{"level": 0}',
        "--generation-config", "vllm",
        "--seed", str(args.seed),
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    environment = {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    }
    if args.profile_dir:
        environment.update({
            "VLLM_TORCH_PROFILER_DIR": str(args.profile_dir.resolve()),
            "VLLM_TORCH_PROFILER_WITH_STACK": "0",
            "VLLM_TORCH_PROFILER_RECORD_SHAPES": "0",
            "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY": "0",
            "VLLM_TORCH_PROFILER_WITH_FLOPS": "0",
        })
    return command, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Optional Hugging Face model commit for repeatable runs")
    parser.add_argument("--gpu", default="0", help="One physical NVIDIA GPU index")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-dir", type=Path, help="Enable server profiler endpoints and save GPU traces here")
    parser.add_argument("--print-command", action="store_true", help="Print only; works without CUDA or vLLM")
    args = parser.parse_args(argv)
    if not args.gpu.isdigit():
        parser.error("--gpu must be one integer index, such as 0")
    if not 1 <= args.port <= 65535 or args.max_model_len < 4097:
        parser.error("Use a valid port and a context length above the 4096 output-token limit")
    command, overrides = build_launch(args)
    print(" ".join(f"{key}={shlex.quote(value)}" for key, value in overrides.items()), shlex.join(command), flush=True)
    if args.print_command:
        return 0
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        parser.error("Install single_gpu/requirements-vllm.txt in the GPU environment first")
    if installed != VLLM_VERSION:
        parser.error(f"This launcher targets vLLM {VLLM_VERSION}; installed version is {installed}")
    environment = os.environ.copy()
    # Prevent an inherited profiler directory from contaminating timing runs.
    environment.pop("VLLM_TORCH_PROFILER_DIR", None)
    environment.update(overrides)
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        gpu_info = subprocess.run(
            ["nvidia-smi", "-i", args.gpu, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        (args.profile_dir / "server_launch.json").write_text(json.dumps({
            "command": command, "environment": overrides, "vllm": installed,
            "python": platform.python_version(), "gpu": gpu_info,
        }, indent=2))
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
