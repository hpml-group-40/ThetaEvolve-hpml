"""Sample GPU and vLLM server counters next to a run; no torch dependency.

Writes a CSV (for Weights & Biases or pandas) and Chrome Trace counter events
that merge_traces() folds into the same Perfetto timeline as the loop spans.
Sampling nvidia-smi and the /metrics page does not touch the model process.
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import urlopen

GPU_FIELDS = ("utilization.gpu", "utilization.memory", "memory.used", "power.draw")
# vLLM 0.11 Prometheus names -> short column names. Missing names are skipped.
VLLM_METRICS = {
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:num_preemptions_total": "preemptions_total",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
}
# Latency histograms: cumulative _sum and _count are sampled; means over any
# interval are (delta sum) / (delta count).
VLLM_HISTOGRAMS = {
    "vllm:time_to_first_token_seconds": "ttft",
    "vllm:time_per_output_token_seconds": "inter_token",
    "vllm:e2e_request_latency_seconds": "e2e",
    "vllm:request_queue_time_seconds": "queue",
    "vllm:request_prefill_time_seconds": "prefill",
    "vllm:request_decode_time_seconds": "decode",
}
for _name, _short in VLLM_HISTOGRAMS.items():
    VLLM_METRICS[_name + "_sum"] = _short + "_seconds_sum"
    VLLM_METRICS[_name + "_count"] = _short + "_count"
METRIC_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-+eE.\d]+|NaN)$")


def parse_metrics(text):
    """Return the selected vLLM gauges/counters from Prometheus text."""
    values = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if match and match.group(1) in VLLM_METRICS:
            try:
                values[VLLM_METRICS[match.group(1)]] = float(match.group(2))
            except ValueError:
                pass
    return values


class Monitor:
    def __init__(self, output_dir, gpu="0", interval_ms=200, metrics_url=None, metrics_interval=1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu, self.interval_ms = gpu, interval_ms
        self.metrics_url, self.metrics_interval = metrics_url, metrics_interval
        self.stop_event = threading.Event()
        self.threads, self.process = [], None
        self.last_metrics_text = None
        self.lock = threading.Lock()
        self.events = (self.output_dir / "client-gpu-monitor.jsonl").open("a")

    def counter(self, name, timestamp_us, values):
        event = {"name": name, "cat": "gpu", "ph": "C", "ts": timestamp_us, "pid": 0, "tid": 0, "args": values}
        with self.lock:
            self.events.write(json.dumps(event) + "\n")
            self.events.flush()

    def _gpu_loop(self):
        self.process = subprocess.Popen(
            ["nvidia-smi", "-i", self.gpu, "--query-gpu=" + ",".join(GPU_FIELDS),
             "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        with (self.output_dir / "gpu_samples.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch_seconds", "gpu_util_percent", "gpu_mem_util_percent", "gpu_mem_used_mib", "gpu_power_watts"])
            for line in self.process.stdout:
                now = time.time()
                try:
                    util, mem_util, mem_used, power = (float(part) for part in line.split(","))
                except ValueError:
                    continue
                writer.writerow([f"{now:.3f}", util, mem_util, mem_used, power])
                stream.flush()
                self.counter("GPU utilization %", now * 1e6, {"util": util})
                self.counter("GPU memory-controller utilization %", now * 1e6, {"util": mem_util})
                self.counter("GPU memory used MiB", now * 1e6, {"used": mem_used})
                self.counter("GPU power W", now * 1e6, {"watts": power})

    def _metrics_loop(self):
        previous = None
        with (self.output_dir / "vllm_metrics.csv").open("w", newline="") as stream:
            columns = ["epoch_seconds"] + sorted(set(VLLM_METRICS.values())) + ["generation_tokens_per_second"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            while not self.stop_event.wait(self.metrics_interval):
                now = time.time()
                try:
                    with urlopen(self.metrics_url, timeout=5) as response:
                        text = response.read().decode()
                    values = parse_metrics(text)
                except Exception:
                    continue
                # Raw first/last pages give exact run deltas, histogram buckets included.
                if previous is None:
                    (self.output_dir / "vllm_metrics_start.txt").write_text(text)
                self.last_metrics_text = text
                rate = None
                if previous and "generation_tokens_total" in values and "generation_tokens_total" in previous[1]:
                    rate = (values["generation_tokens_total"] - previous[1]["generation_tokens_total"]) / (now - previous[0])
                before = previous[1] if previous else {}
                previous = (now, values)
                writer.writerow({"epoch_seconds": f"{now:.3f}", "generation_tokens_per_second": rate, **values})
                stream.flush()
                for key in ("requests_running", "requests_waiting", "kv_cache_usage"):
                    if key in values:
                        self.counter(f"vLLM {key}", now * 1e6, {"value": values[key]})
                if rate is not None:
                    self.counter("vLLM generation tokens/s", now * 1e6, {"value": rate})
                for short, label in (("ttft", "time to first token ms"), ("inter_token", "inter-token latency ms")):
                    count = values.get(short + "_count", 0) - before.get(short + "_count", 0)
                    if before and count > 0:
                        total = values[short + "_seconds_sum"] - before[short + "_seconds_sum"]
                        self.counter(f"vLLM {label}", now * 1e6, {"value": 1000 * total / count})

    def start(self):
        targets = [self._gpu_loop] + ([self._metrics_loop] if self.metrics_url else [])
        self.threads = [threading.Thread(target=target, daemon=True) for target in targets]
        for thread in self.threads:
            thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.metrics_url:
            try:  # one last page after the run's final request has finished
                with urlopen(self.metrics_url, timeout=5) as response:
                    self.last_metrics_text = response.read().decode()
            except Exception:
                pass
            if self.last_metrics_text:
                (self.output_dir / "vllm_metrics_end.txt").write_text(self.last_metrics_text)
        if self.process:
            self.process.terminate()
        for thread in self.threads:
            thread.join(timeout=10)
        with self.lock:
            self.events.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--metrics-url", help="e.g. http://127.0.0.1:8000/metrics")
    args = parser.parse_args(argv)
    monitor = Monitor(args.output, args.gpu, args.interval_ms, args.metrics_url).start()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
