"""Summarize a merged loop trace: where a step's time goes and how busy the GPU is.

Usage: python trace_summary.py <loop_trace.json> [--output summary.json]
Spans nest (iteration > llm_generate > llm_request), so shares do not add to 100%.
"""

import argparse
import json
import statistics
from pathlib import Path


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarize(events):
    spans = [event for event in events if event.get("ph") == "X"]
    gpu = sorted((event["ts"], event["args"]["util"]) for event in events
                 if event.get("ph") == "C" and event["name"] == "GPU utilization %")
    memory = [event["args"]["used"] for event in events
              if event.get("ph") == "C" and event["name"] == "GPU memory used MiB"]
    summary = {"span_count": len(spans), "gpu_sample_count": len(gpu)}
    if not spans:
        return summary
    start = min(event["ts"] for event in spans)
    end = max(event["ts"] + event["dur"] for event in spans)
    wall = (end - start) / 1e6
    summary["traced_wall_seconds"] = wall

    stages = {}
    for event in spans:
        stages.setdefault(event["name"], []).append(event["dur"] / 1e6)
    summary["stages"] = {
        name: {
            "count": len(durations), "total_seconds": sum(durations),
            "share_of_wall": sum(durations) / wall if wall else None,
            "median_seconds": statistics.median(durations),
            "p10_seconds": _percentile(durations, 0.10), "p90_seconds": _percentile(durations, 0.90),
        }
        for name, durations in sorted(stages.items(), key=lambda item: -sum(item[1]))
    }

    requests = [event for event in spans if event["name"] == "llm_request"]
    completion = sum(event["args"].get("completion_tokens") or 0 for event in requests)
    prompt = sum(event["args"].get("prompt_tokens") or 0 for event in requests)
    request_seconds = sum(event["dur"] for event in requests) / 1e6
    summary["llm"] = {
        "requests": len(requests), "prompt_tokens": prompt, "completion_tokens": completion,
        "request_seconds": request_seconds,
        "output_tokens_per_second_while_generating": completion / request_seconds if request_seconds else None,
        "output_tokens_per_second_of_wall": completion / wall if wall else None,
        "truncated_at_max_tokens": sum(event["args"].get("finish_reason") == "length" for event in requests),
    }
    iterations = [event for event in spans if event["name"] == "iteration"]
    evaluations = [event for event in spans if event["name"] == "candidate_evaluation"]
    summary["loop"] = {
        "iterations_started": len(iterations),
        "candidates_evaluated": len(evaluations),
        "candidates_evaluated_per_hour": len(evaluations) / wall * 3600 if wall else None,
    }

    if gpu:
        windows = [(event["ts"], event["ts"] + event["dur"]) for event in requests]
        in_run = [(ts, util) for ts, util in gpu if start <= ts <= end]
        inside = [util for ts, util in in_run if any(low <= ts <= high for low, high in windows)]
        outside = [util for ts, util in in_run if not any(low <= ts <= high for low, high in windows)]
        all_utils = [util for _, util in in_run]
        mean = lambda values: statistics.fmean(values) if values else None
        summary["gpu"] = {
            "mean_utilization_percent": mean(all_utils),
            "mean_utilization_during_llm_requests": mean(inside),
            "mean_utilization_outside_llm_requests": mean(outside),
            "fraction_of_samples_idle_below_5_percent": (
                sum(util < 5 for util in all_utils) / len(all_utils) if all_utils else None),
            "fraction_of_wall_inside_llm_requests": request_seconds / wall if wall else None,
            "peak_memory_used_mib": max(memory) if memory else None,
            "note": "nvidia-smi utilization = share of time any kernel was running; not arithmetic intensity.",
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = summarize(json.loads(args.trace.read_text())["traceEvents"])
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
