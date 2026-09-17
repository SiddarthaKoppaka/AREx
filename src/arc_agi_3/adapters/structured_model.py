"""Validated structured-output adapter with LM-owned repair attempts."""

import json

from pydantic import JsonValue, ValidationError

from arc_agi_3.contracts.decision import (
    AgentContext,
    CognitiveDecision,
    ModelAttempt,
    ModelResponse,
    ModelUsage,
)
from arc_agi_3.trace.canonical import canonical_hash, canonical_json

from .inference import InferenceBackend
from .prompts import decision_prompt


class StructuredOutputError(RuntimeError):
    def __init__(self, attempts: tuple[ModelAttempt, ...]) -> None:
        super().__init__("model did not produce a valid CognitiveDecision")
        self.trace_payload: dict[str, JsonValue] = {
            "attempts": [item.model_dump(mode="json") for item in attempts]
        }


class StructuredModelAdapter:
    def __init__(self, backend: InferenceBackend, max_repairs: int = 1) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        self.backend, self.max_repairs = backend, max_repairs

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {
            "adapter": "structured-model",
            "max_repairs": self.max_repairs,
            "backend": self.backend.metadata,
        }

    @staticmethod
    def _error(error: Exception) -> str:
        if isinstance(error, ValidationError):
            return canonical_json(
                error.errors(
                    include_context=False, include_input=False, include_url=False
                )
            )
        return f"{type(error).__name__}: {error}"

    def decide(self, context: AgentContext) -> ModelResponse:
        attempts: list[ModelAttempt] = []
        input_tokens = output_tokens = latency_ms = 0
        prior: str | None = None
        validation_error: str | None = None
        schema = CognitiveDecision.model_json_schema()
        for number in range(1, self.max_repairs + 2):
            prompt = decision_prompt(
                context, prior_output=prior, validation_error=validation_error
            )
            generated = self.backend.generate(prompt, schema)
            input_tokens += generated.usage.input_tokens
            output_tokens += generated.usage.output_tokens
            latency_ms += generated.usage.latency_ms
            try:
                decision = CognitiveDecision.model_validate(json.loads(generated.text))
            except (json.JSONDecodeError, ValidationError) as error:
                validation_error = self._error(error)
                attempts.append(
                    ModelAttempt(
                        attempt=number,
                        output_hash=canonical_hash(generated.text),
                        valid=False,
                        validation_error=validation_error,
                    )
                )
                prior = generated.text
                continue
            attempts.append(
                ModelAttempt(
                    attempt=number,
                    output_hash=canonical_hash(generated.text),
                    valid=True,
                )
            )
            usage = ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            return ModelResponse(
                decision=decision, usage=usage, attempts=tuple(attempts)
            )
        raise StructuredOutputError(tuple(attempts))
