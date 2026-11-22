import asyncio
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml

from openevolve.config import Config
from openevolve.llm.openai import OpenAILLM


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    request_path = None
    request_payload = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        type(self).request_path = self.path
        type(self).request_payload = json.loads(self.rfile.read(content_length))
        response = {
            "id": "chatcmpl-local-test",
            "object": "chat.completion",
            "created": 0,
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock response"},
                    "finish_reason": "stop",
                }
            ],
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def load_profile(filename):
    with open(CONFIG_DIR / filename, "r") as config_file:
        return Config.from_dict(yaml.safe_load(config_file))


class ProviderConfigTests(unittest.TestCase):
    def test_openrouter_uses_environment_key_and_openai_compatible_base(self):
        config = load_profile("config_anomaly_detection.yaml")
        model = config.llm.models[0]

        self.assertEqual(model.api_base, "https://openrouter.ai/api/v1")
        self.assertIsNone(model.api_key)
        self.assertFalse(config.evaluator.collect_runtime_environments)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-key"}):
            client = OpenAILLM(model).client
        self.assertEqual(client.api_key, "unit-test-key")

    def test_vllm_profile_sets_base_url_and_served_model_placeholder(self):
        config = load_profile("config_anomaly_detection_vllm.yaml")
        model = config.llm.models[0]

        self.assertEqual(model.api_base, "http://localhost:8000/v1")
        self.assertEqual(model.name, "REPLACE_WITH_VLLM_SERVED_MODEL")
        self.assertEqual(model.api_key, "EMPTY")
        self.assertFalse(config.evaluator.collect_runtime_environments)

    def test_openai_client_uses_chat_completions_endpoint(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAICompatibleHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        OpenAICompatibleHandler.request_path = None
        OpenAICompatibleHandler.request_payload = None
        try:
            config = load_profile("config_anomaly_detection_vllm.yaml")
            model = config.llm.models[0]
            model.name = "mock-model"
            model.api_base = f"http://127.0.0.1:{server.server_port}/v1"
            model.api_key = "local-test-key"
            client = OpenAILLM(model)

            with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost"}):
                response = asyncio.run(
                    client.generate_with_context(
                        "system instructions",
                        [{"role": "user", "content": "test prompt"}],
                        seed=17,
                    )
                )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

        self.assertEqual(response, "mock response")
        self.assertEqual(OpenAICompatibleHandler.request_path, "/v1/chat/completions")
        self.assertEqual(OpenAICompatibleHandler.request_payload["model"], "mock-model")
        self.assertEqual(OpenAICompatibleHandler.request_payload["seed"], 17)
        self.assertEqual(
            OpenAICompatibleHandler.request_payload["messages"],
            [
                {"role": "system", "content": "system instructions"},
                {"role": "user", "content": "test prompt"},
            ],
        )


if __name__ == "__main__":
    unittest.main()