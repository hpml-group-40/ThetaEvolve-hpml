"""CPU-only checks for traces and vLLM profiler lifecycle; no model download."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from openevolve.config import LLMConfig, LLMModelConfig
from openevolve.llm.openai import OpenAILLM
from openevolve.utils.trace_utils import (
    _profile_endpoint, merge_traces, profile_vllm_request, trace_span,
)


class ProfilingTests(unittest.TestCase):
    def test_merge_retains_worker_spans_and_errors_after_a_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, HPML_TRACE_DIR=directory)
            subprocess.run([
                sys.executable, "-c",
                "from openevolve.utils.trace_utils import trace_span\n"
                "with trace_span('worker', iteration=1): pass\n",
            ], env=environment, check=True)
            with patch.dict(os.environ, {"HPML_TRACE_DIR": directory}):
                with self.assertRaises(ValueError):
                    with trace_span("failed_evaluation"):
                        raise ValueError("synthetic failure")
            with (Path(directory) / f"client-{os.getpid()}.jsonl").open("a") as stream:
                stream.write('{"incomplete":')
            destination = Path(directory) / "loop.json"
            events = merge_traces(directory, destination)
            self.assertEqual({event["name"] for event in events}, {"worker", "failed_evaluation"})
            self.assertEqual(len({event["pid"] for event in events}), 2)
            failed = next(event for event in events if event["name"] == "failed_evaluation")
            self.assertEqual(failed["args"]["error"], "ValueError")
            self.assertEqual(json.loads(destination.read_text())["traceEvents"], events)

    def test_client_profiles_once_and_does_not_change_chat_responses(self):
        paths = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                paths.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                payload = {"choices": [{"message": {"role": "assistant", "content": "a diff"}}]}
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = f"http://127.0.0.1:{server.server_port}/v1"
                config = LLMConfig(models=[LLMModelConfig(name="mock")], api_base=base, api_key="EMPTY", retries=0)
                client = OpenAILLM(config.models[0])
                with patch.dict(os.environ, {"HPML_TRACE_DIR": directory, "HPML_VLLM_PROFILE": "1", "NO_PROXY": "127.0.0.1,localhost"}):
                    for _ in range(2):
                        response = asyncio.run(client.generate("edit this code"))
                        self.assertEqual(response, "a diff")
                self.assertEqual(paths, ["/start_profile", "/v1/chat/completions", "/stop_profile", "/v1/chat/completions"])
                status = json.loads((Path(directory) / "gpu_profile_status.json").read_text())
                self.assertEqual(status["status"], "complete")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_generation_error_stops_the_gpu_profiler(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"HPML_TRACE_DIR": directory, "HPML_VLLM_PROFILE": "1"}):
                with patch("openevolve.utils.trace_utils._profile_endpoint") as endpoint:
                    with self.assertRaises(RuntimeError):
                        with profile_vllm_request("http://localhost:8000/v1", "EMPTY"):
                            raise RuntimeError("generation failed")
                self.assertEqual([call.args[1] for call in endpoint.call_args_list], ["start_profile", "stop_profile"])
                status = json.loads((Path(directory) / "gpu_profile_status.json").read_text())
                self.assertEqual(status["status"], "error")

    def test_opt_out_makes_no_profiler_calls(self):
        with patch.dict(os.environ, {"HPML_VLLM_PROFILE": "0"}):
            with patch("openevolve.utils.trace_utils._profile_endpoint") as endpoint:
                with profile_vllm_request("https://openrouter.ai/api/v1", "fake-key"):
                    pass
            endpoint.assert_not_called()

    def test_remote_providers_cannot_receive_gpu_profiler_calls(self):
        with self.assertRaises(ValueError):
            _profile_endpoint("https://openrouter.ai/api/v1", "start_profile", "fake-key")


if __name__ == "__main__":
    unittest.main()
