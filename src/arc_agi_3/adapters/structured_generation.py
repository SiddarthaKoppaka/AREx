"""The one bounded primary-plus-repair generation engine.

Every structured output in this project — CognitiveDecision, SpecialistReport,
SpecialistReview — goes through this single function. Only attempt 1 ("primary")
sees its full prompt; every repair uses `build_repair_prompt` instead: just
the malformed text, the schema, and why it failed. `max_repairs` bounds the
total, so one bad parse costs at most `max_repairs + 1` backend
generations, never dozens, regardless of what is being generated.
"""

import json
from collections.abc import Callable
from functools import partial
from typing import Any

from pydantic import BaseModel, ValidationError

from arc_agi_3.contracts.decision import ModelAttempt, ModelUsage, PromptReport
from arc_agi_3.contracts.generation import ErrorCategory

from .context_limits import guard_prompt_tokens
from .generation_attempts import accumulate, record_attempt
from .generation_errors import GenerationBackendError, classify
from .inference import InferenceBackend
from .repair_prompt import build_repair_prompt
from .structured_output import (
    StructuredOutputError,
    parse_json_object,
    validation_error_text,
)


def run_structured[T: BaseModel](
    backend: InferenceBackend,
    schema: dict[str, Any],
    output_model: type[T],
    primary: Callable[[], tuple[str, PromptReport | None]],
    max_repairs: int,
    *,
    persist_invalid_output: bool,
    preview_chars: int,
    turn: int,
    validate_extra: Callable[[T], None] | None = None,
) -> tuple[T, ModelUsage, tuple[ModelAttempt, ...], PromptReport | None]:
    attempts: list[ModelAttempt] = []
    totals = ModelUsage()
    malformed = error_text = ""
    category: ErrorCategory = "unknown_backend_failure"
    record = partial(
        record_attempt,
        persist_invalid_output=persist_invalid_output,
        preview_chars=preview_chars,
    )
    report: PromptReport | None = None
    for number in range(1, max_repairs + 2):
        if number == 1:
            prompt, report = primary()
        else:
            prompt = build_repair_prompt(
                category, error_text, malformed, target_type=output_model.__name__
            )
            guard_prompt_tokens(backend, schema, prompt, context=f"turn={turn};repair")
        try:
            generated = backend.generate(prompt, schema)
        except GenerationBackendError as error:
            totals = accumulate(totals, error.usage)
            category, error_text, malformed = "unknown_backend_failure", str(error), ""
            attempts.append(
                record(number, "", category, error_text, False, error.usage)
            )
            continue
        totals = accumulate(totals, generated.usage)
        try:
            parsed = output_model.model_validate(parse_json_object(generated.text))
            if validate_extra is not None:
                validate_extra(parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            category = classify(error, generated.text)
            error_text, malformed = validation_error_text(error), generated.text
            attempts.append(
                record(
                    number,
                    generated.text,
                    category,
                    error_text,
                    False,
                    generated.usage,
                )
            )
            continue
        attempts.append(
            record(number, generated.text, None, None, True, generated.usage)
        )
        return parsed, totals, tuple(attempts), report
    raise StructuredOutputError(tuple(attempts), totals, output_model.__name__)
