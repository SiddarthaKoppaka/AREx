"""Parsing and failure metadata for structured model output."""

import json

from pydantic import JsonValue, ValidationError

from arc_agi_3.contracts.decision import ModelAttempt, ModelUsage
from arc_agi_3.trace.canonical import canonical_json


def validation_error_text(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return canonical_json(
            error.errors(include_context=False, include_input=False, include_url=False)
        )
    return f"{type(error).__name__}: {error}"


def parse_json_object(text: str) -> dict[str, object]:
    """Extract the first valid JSON object from raw or wrapped model output."""
    stripped = text.strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise json.JSONDecodeError(
        "No valid JSON object found in model output", stripped, 0
    )


class StructuredOutputError(RuntimeError):
    def __init__(self, attempts: tuple[ModelAttempt, ...], usage: ModelUsage) -> None:
        super().__init__("model did not produce a valid CognitiveDecision")
        self.usage = usage
        self.trace_payload: dict[str, JsonValue] = {
            "attempts": [
                item.model_dump(mode="json", exclude_none=True) for item in attempts
            ],
            "generation_attempts": len(attempts),
            "usage": usage.model_dump(mode="json"),
        }
