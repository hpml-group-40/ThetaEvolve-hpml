"""Opt-in CPU spans and local vLLM GPU profiling; no torch dependency in the client."""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name, **fields):
    """Write completed Chrome Trace events, one append-only file per process.

    Epoch microseconds let processes on this host share a timeline. Durations use
    a monotonic clock. Partial runs retain all spans completed before shutdown.
    """
    directory = os.environ.get("HPML_TRACE_DIR")
    if not directory:
        yield
        return
    timestamp_us = time.time_ns() / 1000
    start_ns = time.perf_counter_ns()
    try:
        yield
    except BaseException as error:
        fields = dict(fields, error=type(error).__name__)
        raise
    finally:
        event = {
            "name": name,
            "cat": "openevolve",
            "ph": "X",
            "ts": timestamp_us,
            "dur": (time.perf_counter_ns() - start_ns) / 1000,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "args": fields,
        }
        try:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            # One OS append per event; threads never share a buffered writer.
            descriptor = os.open(
                root / f"client-{os.getpid()}.jsonl",
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, (json.dumps(event) + "\n").encode())
            finally:
                os.close(descriptor)
        except OSError:
            logger.exception("Could not write HPML trace event")


def traced(name):
    """Trace a synchronous function without changing its call signature."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with trace_span(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def merge_traces(directory, destination):
    """Convert per-process JSONL into a Perfetto-readable Chrome Trace file."""
    events = []
    for path in sorted(Path(directory).glob("client-*.jsonl")):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A force-killed writer may leave one incomplete final record.
                if index != len(lines) - 1:
                    raise
                logger.warning("Ignoring incomplete last event in %s", path)
    events.sort(key=lambda event: event["ts"])
    Path(destination).write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}))
    return events


def record_target_reached(iteration, score):
    """Record the controller's threshold observation independently of profiling.

    The launcher and controller run on the same host and share a monotonic clock.
    This excludes model startup/warmup and avoids counting trace export as search.
    """
    with trace_span("target_reached", iteration=iteration, dev_f1=score):
        path = os.environ.get("HPML_TARGET_STATUS_PATH")
        start = os.environ.get("HPML_RUN_START_NS")
        if path and start:
            status = {
                "iteration": iteration, "dev_f1": score,
                "elapsed_seconds": (time.perf_counter_ns() - int(start)) / 1e9,
            }
            Path(path).write_text(json.dumps(status))


def _profile_endpoint(api_base, endpoint, api_key):
    base = urlsplit(api_base)
    if (
        base.scheme != "http"
        or base.hostname not in {"127.0.0.1", "localhost", "::1"}
        or base.path.rstrip("/") != "/v1"
        or base.query
        or base.fragment
        or base.username
        or base.password
    ):
        raise ValueError("GPU profiling requires a local vLLM http://localhost:<port>/v1 URL")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = Request(f"{base.scheme}://{base.netloc}/{endpoint}", headers=headers, method="POST")
    with urlopen(request, timeout=120) as response:
        response.read()


@contextmanager
def profile_vllm_request(api_base, api_key):
    """Profile only the first LLM call across all client workers in this run.

    This scope runs *inside* the API thread so stop_profile still happens when
    the outer async wait is cancelled. The single-GPU profile launcher uses one
    worker and also stops the server profiler after a hard client timeout.
    """
    if os.environ.get("HPML_VLLM_PROFILE") != "1":
        yield
        return
    directory = os.environ.get("HPML_TRACE_DIR")
    if not directory:
        raise ValueError("HPML_VLLM_PROFILE requires HPML_TRACE_DIR")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(root / "gpu-profile.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        yield
        return
    os.close(descriptor)

    status = {"status": "starting", "api_base": api_base, "pid": os.getpid()}
    status_path = root / "gpu_profile_status.json"
    started = False
    try:
        with trace_span("gpu_profiler_start"):
            _profile_endpoint(api_base, "start_profile", api_key)
        started = True
        status.update(status="recording", started=True, stopped=False)
        status_path.write_text(json.dumps(status))
        try:
            yield
        finally:
            with trace_span("gpu_profiler_stop"):
                _profile_endpoint(api_base, "stop_profile", api_key)
            status["stopped"] = True
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="error", error=str(error), started=started)
        raise
    finally:
        status_path.write_text(json.dumps(status))
