"""The bounded primary-plus-repair generation loop.

Only attempt 1 ("primary") sees the working set. Every repair attempt uses
`build_repair_prompt` instead: just the malformed text, the schema, and why
it failed. `max_repairs` is enforced by the caller, so one bad parse costs
at most `max_repairs + 1` backend generations, never dozens.
"""

import json
from functools import partial

from pydantic import ValidationError

from arc_agi_3.contracts.decision import (
    AgentContext,
    CognitiveDecision,
    ModelAttempt,
    ModelResponse,
    ModelUsage,
)
from arc_agi_3.contracts.generation import ErrorCategory

from .context_compaction import compact_with_report
from .context_limits import guard_prompt_tokens
from .decision_schema import decision_schema
from .generation_attempts import accumulate, record_attempt
from .generation_errors import GenerationBackendError, classify
from .inference import InferenceBackend
from .prompts import decision_prompt
from .repair_prompt import build_repair_prompt
from .structured_output import (
    StructuredOutputError,
    parse_json_object,
    validation_error_text,
)


def run_generation(
    backend: InferenceBackend,
    context: AgentContext,
    max_repairs: int,
    *,
    persist_invalid_output: bool,
    preview_chars: int,
) -> ModelResponse:
    attempts: list[ModelAttempt] = []
    totals = ModelUsage()
    malformed = error_text = ""
    category: ErrorCategory = "unknown_backend_failure"
    schema = decision_schema()
    record = partial(
        record_attempt,
        persist_invalid_output=persist_invalid_output,
        preview_chars=preview_chars,
    )
    report = None
    for number in range(1, max_repairs + 2):
        if number == 1:
            projected, report = compact_with_report(context, backend, schema)
            prompt = decision_prompt(projected)
        else:
            prompt = build_repair_prompt(category, error_text, malformed)
            guard_prompt_tokens(
                backend, schema, prompt, context=f"turn={context.turn};repair"
            )
        try:
            generated = backend.generate(prompt, schema)
        except GenerationBackendError as error:
            totals = accumulate(totals, error.usage)
            category = "unknown_backend_failure"
            error_text, malformed = str(error), ""
            attempts.append(record(number, "", category, error_text, False))
            continue
        totals = accumulate(totals, generated.usage)
        try:
            decision = CognitiveDecision.model_validate(
                parse_json_object(generated.text)
            )
        except (json.JSONDecodeError, ValidationError) as error:
            category = classify(error, generated.text)
            error_text, malformed = validation_error_text(error), generated.text
            attempts.append(record(number, generated.text, category, error_text, False))
            continue
        attempts.append(
            record(number, generated.text, None, None, True, generated.usage)
        )
        return ModelResponse(
            decision=decision, usage=totals, attempts=tuple(attempts), prompt=report
        )
    raise StructuredOutputError(tuple(attempts), totals)
