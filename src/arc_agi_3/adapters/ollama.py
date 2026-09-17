"""Loopback Ollama backend using schema-constrained chat generation."""

import json
from typing import Any
from urllib.request import Request, urlopen

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.trace.canonical import canonical_json

from .inference import BackendGeneration


class OllamaConfig(Contract):
    model: str = "qwen3.5:9b"
    model_digest: str = "unresolved"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int = 0
    num_ctx: int = Field(default=32768, gt=0)
    timeout_seconds: float = Field(default=300.0, gt=0)
    keep_alive: str = "10m"


class OllamaBackend:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.config.model,
            "model_digest": self.config.model_digest,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "num_ctx": self.config.num_ctx,
        }

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendGeneration:
        grounded = prompt + "\nJSON_SCHEMA:\n" + canonical_json(json_schema)
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": grounded}],
            "format": json_schema,
            "stream": False,
            "think": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_ctx": self.config.num_ctx,
            },
        }
        request = Request(
            self.config.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            result = json.loads(response.read())
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama response lacks message.content")
        return BackendGeneration(
            text=message["content"],
            usage=ModelUsage(
                input_tokens=_integer(result.get("prompt_eval_count")),
                output_tokens=_integer(result.get("eval_count")),
                latency_ms=_integer(result.get("total_duration")) // 1_000_000,
            ),
        )


def _integer(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0
