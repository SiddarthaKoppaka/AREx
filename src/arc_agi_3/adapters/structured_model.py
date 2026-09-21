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
from arc_agi_3.trace.canonical import canonical_hash

from .context_compaction import compact_for_backend
from .inference import InferenceBackend
from .prompts import decision_prompt
from .structured_output import (
    StructuredOutputError,
    parse_json_object,
    validation_error_text,
)


class StructuredModelAdapter:
    def __init__(
        self,
        backend: InferenceBackend,
        max_repairs: int = 1,
        persist_invalid_output: bool = False,
    ) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        self.backend, self.max_repairs = backend, max_repairs
        self.persist_invalid_output = persist_invalid_output

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {
            "adapter": "structured-model",
            "max_repairs": self.max_repairs,
            "persist_invalid_output": self.persist_invalid_output,
            "backend": self.backend.metadata,
        }

    def decide(self, context: AgentContext) -> ModelResponse:
        attempts: list[ModelAttempt] = []
        input_tokens = output_tokens = latency_ms = 0
        prior: str | None = None
        validation_error: str | None = None
        usage = ModelUsage()
        schema = CognitiveDecision.model_json_schema()
        for number in range(1, self.max_repairs + 2):
            projected = compact_for_backend(
                context,
                self.backend,
                schema,
                prior_output=prior,
                validation_error=validation_error,
            )
            prompt = decision_prompt(
                projected, prior_output=prior, validation_error=validation_error
            )
            generated = self.backend.generate(prompt, schema)
            input_tokens += generated.usage.input_tokens
            output_tokens += generated.usage.output_tokens
            latency_ms += generated.usage.latency_ms
            usage = ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            try:
                decision = CognitiveDecision.model_validate(
                    parse_json_object(generated.text)
                )
            except (json.JSONDecodeError, ValidationError) as error:
                validation_error = validation_error_text(error)
                attempts.append(
                    ModelAttempt(
                        attempt=number,
                        output_hash=canonical_hash(generated.text),
                        valid=False,
                        validation_error=validation_error,
                        output_preview=generated.text[:512]
                        if self.persist_invalid_output
                        else None,
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
            return ModelResponse(
                decision=decision, usage=usage, attempts=tuple(attempts)
            )
        raise StructuredOutputError(tuple(attempts), usage)
