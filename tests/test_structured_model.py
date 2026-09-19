"""Structured parsing and non-semantic repair behavior."""

import json

import pytest
from pydantic import JsonValue

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.adapters.structured_model import (
    StructuredModelAdapter,
    StructuredOutputError,
    parse_json_object,
)
from arc_agi_3.contracts.decision import AgentContext, ModelUsage
from arc_agi_3.testing import FakeLineEnvironment, successful_script


class Backend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "test", "model": "scripted-text"}

    def generate(
        self, prompt: str, json_schema: dict[str, object]
    ) -> BackendGeneration:
        self.prompts.append(prompt)
        return BackendGeneration(
            text=self.outputs.pop(0),
            usage=ModelUsage(input_tokens=2, output_tokens=3, latency_ms=4),
        )


def context() -> AgentContext:
    return AgentContext(
        turn=1,
        observation=FakeLineEnvironment().reset(),
        budget={},
    )


def test_invalid_json_is_repaired_by_the_model() -> None:
    valid = json.dumps(successful_script()[0].model_dump(mode="json"))
    backend = Backend(["{}", valid])
    response = StructuredModelAdapter(backend, max_repairs=1).decide(context())
    assert [attempt.valid for attempt in response.attempts] == [False, True]
    assert response.usage == ModelUsage(input_tokens=4, output_tokens=6, latency_ms=8)
    assert "Preserve your decision semantics" in backend.prompts[1]
    assert "private chain-of-thought" in backend.prompts[0]
    assert "execute mode must set exactly one" in backend.prompts[0]
    assert "Do not use Markdown code fences" in backend.prompts[0]


@pytest.mark.parametrize(
    "output",
    [
        '{"status": "ok"}',
        '```json\n{"status": "ok"}\n```',
        'Result: {broken} then {"status": "ok"}',
    ],
)
def test_parse_json_object_accepts_wrapped_output(output: str) -> None:
    assert parse_json_object(output) == {"status": "ok"}


def test_parse_json_object_rejects_missing_object() -> None:
    with pytest.raises(json.JSONDecodeError, match="No valid JSON object"):
        parse_json_object("```json\n[]\n```")


def test_fenced_repair_keeps_cross_field_validation_strict() -> None:
    invalid = successful_script()[0].model_dump(mode="json")
    invalid["mode"] = "plan"
    valid = json.dumps(successful_script()[0].model_dump(mode="json"))
    backend = Backend([json.dumps(invalid), f"```json\n{valid}\n```"])
    response = StructuredModelAdapter(backend).decide(context())
    assert [attempt.valid for attempt in response.attempts] == [False, True]
    assert "only execute decisions may authorize actions" in backend.prompts[1]
    assert response.usage == ModelUsage(input_tokens=4, output_tokens=6, latency_ms=8)


def test_cross_field_error_is_json_safe_for_repair() -> None:
    invalid = json.dumps({"assessment": "test", "intent": "act", "mode": "execute"})
    valid = json.dumps(successful_script()[0].model_dump(mode="json"))
    backend = Backend([invalid, valid])
    response = StructuredModelAdapter(backend).decide(context())
    assert [attempt.valid for attempt in response.attempts] == [False, True]
    assert "Value error" in backend.prompts[1]


def test_exhausted_repairs_expose_hashed_attempt_metadata() -> None:
    with pytest.raises(StructuredOutputError) as caught:
        StructuredModelAdapter(Backend(["{}"]), max_repairs=0).decide(context())
    attempts = caught.value.trace_payload["attempts"]
    assert isinstance(attempts, list)
    assert "{}" not in str(attempts)
