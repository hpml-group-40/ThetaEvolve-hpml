"""Summarize a merged loop trace: where a step's time goes and how busy the GPU is.

Usage: python trace_summary.py <loop_trace.json> [--output summary.json]
Spans nest (iteration > llm_generate > llm_request), so shares do not add to 100%.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

HISTOGRAMS = {
    "time_to_first_token": "vllm:time_to_first_token_seconds",
    "inter_token_latency": "vllm:time_per_output_token_seconds",
    "end_to_end_latency": "vllm:e2e_request_latency_seconds",
    "queue_time": "vllm:request_queue_time_seconds",
    "prefill_time": "vllm:request_prefill_time_seconds",
    "decode_time": "vllm:request_decode_time_seconds",
}
SAMPLE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+eE.\d]+)$")


def _histograms(text):
    """{metric: {"sum": x, "count": n, "buckets": [(upper_bound, cumulative_count), ...]}}"""
    found = {}
    for line in (text or "").splitlines():
        match = SAMPLE.match(line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        for suffix in ("_sum", "_count", "_bucket"):
            base = name[: -len(suffix)]
            if name.endswith(suffix) and base in HISTOGRAMS.values():
                entry = found.setdefault(base, {"sum": 0.0, "count": 0.0, "buckets": []})
                if suffix == "_bucket":
                    bound = re.search(r'le="([^"]+)"', labels).group(1)
                    entry["buckets"].append((float("inf") if bound == "+Inf" else float(bound), value))
                else:
                    entry[suffix[1:]] = value
    return found


def _quantile(buckets, count, fraction):
    """Linear interpolation inside the bucket, as Prometheus does. Accuracy = bucket width."""
    rank, lower, below = fraction * count, 0.0, 0.0
    for index, (upper, cumulative) in enumerate(buckets):
        if cumulative >= rank:
            if index == 0:
                return None  # below the histogram's resolution; use the mean
            if upper == float("inf") or cumulative == below:
                return lower
            return lower + (upper - lower) * (rank - below) / (cumulative - below)
        lower, below = upper, cumulative
    return None


def serving_latency(start_text, end_text):
    """Per-run latency statistics from the first and last vLLM /metrics pages."""
    start, end, result = _histograms(start_text), _histograms(end_text), {}
    for label, metric in HISTOGRAMS.items():
        if metric not in end:
            continue
        first = start.get(metric, {"sum": 0.0, "count": 0.0, "buckets": []})
        count = end[metric]["count"] - first["count"]
        if count <= 0:
            continue
        earlier = dict(first["buckets"])
        buckets = [(bound, value - earlier.get(bound, 0.0)) for bound, value in sorted(end[metric]["buckets"])]
        total = end[metric]["sum"] - first["sum"]
        result[label] = {
            "count": count, "total_seconds": total, "mean_seconds": total / count,
            "p50_seconds": _quantile(buckets, count, 0.50), "p90_seconds": _quantile(buckets, count, 0.90),
        }
    if result:
        result["note"] = "Means are exact; p50/p90 are interpolated within vLLM's histogram buckets."
    return result


def time_to_target(curve, target):
    """First elapsed second at which the best search F1 reached the target, else None."""
    return next((point["elapsed_seconds"] for point in curve if point["best_f1"] >= target), None)


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarize(events, trace_dir=None, weights_gib=None, bandwidth_gb_per_s=300.0, bin_seconds=60):
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

    results = sorted((event for event in spans if event["name"] == "iteration_result"), key=lambda e: e["ts"])
    graded = [event["args"] for event in results if event["args"].get("outcome") == "graded"]
    discarded = [event["args"] for event in results if event["args"].get("outcome") == "discarded"]
    reasons = {}
    for item in discarded:
        reasons[item.get("reason")] = reasons.get(item.get("reason"), 0) + 1
    summary["programs"] = {
        "llm_requests": len(requests),
        "replies_discarded": len(discarded), "discard_reasons": reasons,
        "graded": len(graded),
        "valid": sum(bool(item.get("valid")) for item in graded),
        "failed_to_run": sum(not item.get("valid") for item in graded),
        "improved_on_parent": sum(bool(item.get("improved_on_parent")) for item in graded),
        "distinct_valid_programs": len({item.get("code_hash") for item in graded if item.get("valid")}),
        "valid_programs_per_hour": sum(bool(i.get("valid")) for i in graded) / wall * 3600 if wall else None,
    }
    curve, best = [], None
    for event in results:
        score = event["args"].get("best_score")
        if score is not None and (best is None or score > best):
            best = score
            elapsed = event["args"].get("elapsed_seconds")
            curve.append({
                "elapsed_seconds": elapsed if elapsed is not None else (event["ts"] - start) / 1e6,
                "best_f1": score, "iteration": event["args"].get("iteration"),
            })
    summary["best_f1_curve"] = curve
    summary["final_best_f1"] = best

    # Minute-by-minute view, e.g. to compare the first minute with steady state.
    bins = {}
    for event in requests:
        index = int((event["ts"] + event["dur"] - start) / 1e6 // bin_seconds)
        row = bins.setdefault(index, {"requests": 0, "completion_tokens": 0, "request_seconds": 0.0})
        row["requests"] += 1
        row["completion_tokens"] += event["args"].get("completion_tokens") or 0
        row["request_seconds"] += event["dur"] / 1e6
    for ts, util in gpu:
        index = int((ts - start) / 1e6 // bin_seconds)
        if index in bins:
            bins[index].setdefault("gpu_utils", []).append(util)
    metric_rows = []
    if trace_dir and (Path(trace_dir) / "vllm_metrics.csv").exists():
        with (Path(trace_dir) / "vllm_metrics.csv").open() as stream:
            metric_rows = [row for row in csv.DictReader(stream) if row.get("ttft_count")]
    number = lambda row, key: float(row[key]) if row.get(key) not in (None, "") else 0.0
    timeline = []
    for index in sorted(bins):
        row = bins[index]
        entry = {
            "from_second": index * bin_seconds, "requests_finished": row["requests"],
            "completion_tokens": row["completion_tokens"],
            "output_tokens_per_second_while_generating": (
                row["completion_tokens"] / row["request_seconds"] if row["request_seconds"] else None),
            "mean_gpu_utilization_percent": statistics.fmean(row["gpu_utils"]) if row.get("gpu_utils") else None,
        }
        inside = [m for m in metric_rows
                  if index * bin_seconds <= float(m["epoch_seconds"]) - start / 1e6 < (index + 1) * bin_seconds]
        if len(inside) >= 2:
            for short, label in (("ttft", "mean_time_to_first_token_seconds"), ("inter_token", "mean_inter_token_latency_seconds")):
                count = number(inside[-1], short + "_count") - number(inside[0], short + "_count")
                total = number(inside[-1], short + "_seconds_sum") - number(inside[0], short + "_seconds_sum")
                entry[label] = total / count if count > 0 else None
        timeline.append(entry)
    summary["timeline"] = timeline

    if trace_dir:
        read = lambda name: (Path(trace_dir) / name).read_text() if (Path(trace_dir) / name).exists() else None
        summary["serving_latency"] = serving_latency(read("vllm_metrics_start.txt"), read("vllm_metrics_end.txt"))
        step = (summary["serving_latency"].get("inter_token_latency") or {}).get("mean_seconds")
        decode = (summary["serving_latency"].get("decode_time") or {}).get("total_seconds")
        if weights_gib and step:
            weights_gb = weights_gib * 2**30 / 1e9
            ceiling = bandwidth_gb_per_s / weights_gb
            summary["hardware_limit"] = {
                "weights_gb": weights_gb, "assumed_memory_bandwidth_gb_per_s": bandwidth_gb_per_s,
                "decode_step_seconds": step,
                "weight_bytes_moved_gb_per_s": weights_gb / step,
                "fraction_of_memory_bandwidth": weights_gb / step / bandwidth_gb_per_s,
                "single_sequence_ceiling_tokens_per_s": ceiling,
                "tokens_per_decode_step": completion / (decode / step) if decode else None,
                "decode_tokens_per_s": completion / decode if decode else None,
                "note": "One decode step reads all weights once. Batching raises tokens per step; "
                        "smaller weights shorten the step. KV-cache reads are not counted.",
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
    parser.add_argument("--trace-dir", type=Path, help="Run's traces/ directory (vLLM metrics pages and CSV)")
    parser.add_argument("--weights-gib", type=float, help="Model weight size, from the server log")
    parser.add_argument("--bandwidth-gb-per-s", type=float, default=300.0, help="GPU memory bandwidth (L4: 300)")
    parser.add_argument("--target-f1", type=float, help="Also print the time this run took to reach the target")
    args = parser.parse_args(argv)
    summary = summarize(json.loads(args.trace.read_text())["traceEvents"], args.trace_dir,
                        args.weights_gib, args.bandwidth_gb_per_s)
    if args.target_f1 is not None:
        summary["time_to_target"] = {"target_f1": args.target_f1,
                                     "seconds": time_to_target(summary["best_f1_curve"], args.target_f1)}
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
