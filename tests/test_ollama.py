"""Ollama request/response mechanics stay behind the inference protocol."""

import json
from typing import Any

from arc_agi_3.adapters.ollama import OllamaBackend, OllamaConfig


class Response:
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "message": {"content": '{"mode":"stop"}'},
                "prompt_eval_count": 12,
                "eval_count": 4,
                "total_duration": 9_000_000,
            }
        ).encode()


def test_backend_requests_schema_and_records_usage(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, Any] = {}

    def open_request(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("arc_agi_3.adapters.ollama.urlopen", open_request)
    config = OllamaConfig(model="qwen3.5:9b", model_digest="digest", seed=7)
    generated = OllamaBackend(config).generate("choose", {"type": "object"})
    payload = captured["payload"]
    assert payload["format"] == {"type": "object"}
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0
    assert "JSON_SCHEMA" in payload["messages"][0]["content"]
    assert generated.usage.input_tokens == 12
    assert generated.usage.output_tokens == 4
    assert generated.usage.latency_ms == 9
    assert OllamaBackend(config).metadata["model_digest"] == "digest"
