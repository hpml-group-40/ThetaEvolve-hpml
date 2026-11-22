"""Summarize a vLLM torch-profiler trace: how busy the GPU was and on what.

Usage: python analyze_kernel_trace.py <dir or *.pt.trace.json.gz> [--output kernel_summary.json]
"""

import argparse
import gzip
import json
from pathlib import Path

FAMILIES = (
    ("matrix_times_vector_gemv", ("gemv",)),
    ("matrix_times_matrix_gemm", ("gemm",)),
    ("attention", ("flash", "attention")),
    ("memory_copy", ("memcpy", "memset")),
    ("activation_and_norm", ("act_and_mul", "rms_norm", "norm", "silu")),
)


def family(name):
    lowered = name.lower()
    return next((label for label, keys in FAMILIES if any(key in lowered for key in keys)), "other")


def analyze(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu = sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    if not gpu:
        return {"file": str(path), "gpu_events": 0}
    start, end = gpu[0]["ts"], max(e["ts"] + e["dur"] for e in gpu)
    busy, low, high = 0.0, gpu[0]["ts"], gpu[0]["ts"] + gpu[0]["dur"]
    for event in gpu[1:]:
        if event["ts"] > high:
            busy, low, high = busy + high - low, event["ts"], event["ts"] + event["dur"]
        else:
            high = max(high, event["ts"] + event["dur"])
    busy += high - low
    by_family, by_name = {}, {}
    for event in gpu:
        by_family[family(event["name"])] = by_family.get(family(event["name"]), 0.0) + event["dur"]
        by_name[event["name"][:90]] = by_name.get(event["name"][:90], 0.0) + event["dur"]
    total = sum(by_family.values())
    launches = [e for e in events if e.get("cat") == "cuda_runtime" and e["name"] == "cudaLaunchKernel"]
    window = (end - start) / 1e6
    return {
        "file": str(path), "window_seconds": window, "gpu_events": len(gpu),
        "gpu_busy_fraction": busy / (end - start),
        "gpu_idle_between_kernels_fraction": 1 - busy / (end - start),
        "kernel_launches_per_second": len(launches) / window if window else None,
        "cpu_seconds_in_kernel_launch_calls": sum(e["dur"] for e in launches) / 1e6,
        "kernel_time_share_by_family": {k: v / total for k, v in sorted(by_family.items(), key=lambda i: -i[1])},
        "top_kernels": [{"name": k, "share": v / total} for k, v in sorted(by_name.items(), key=lambda i: -i[1])[:8]],
        "note": "gemv = one token's vector per weight matrix (batch of 1); a batched server shifts this to gemm. "
                "Idle-between-kernels bounds what CUDA graphs can remove.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    files = sorted(args.path.glob("*.pt.trace.json*")) if args.path.is_dir() else [args.path]
    # The engine-core process holds the CUDA events; the API process trace is tiny.
    summary = max((analyze(path) for path in files), key=lambda item: item.get("gpu_events", 0))
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
