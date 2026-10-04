"""Build one bounded, telemetry-rich ModelAttempt record per generation."""

from arc_agi_3.contracts.decision import ModelAttempt, ModelUsage
from arc_agi_3.contracts.generation import ErrorCategory
from arc_agi_3.trace.canonical import canonical_hash

from .generation_errors import has_json_candidate


def accumulate(totals: ModelUsage, usage: ModelUsage) -> ModelUsage:
    return ModelUsage(
        input_tokens=totals.input_tokens + usage.input_tokens,
        output_tokens=totals.output_tokens + usage.output_tokens,
        latency_ms=totals.latency_ms + usage.latency_ms,
    )


def record_attempt(
    number: int,
    text: str,
    category: ErrorCategory | None,
    error_text: str | None,
    valid: bool,
    usage: ModelUsage | None = None,
    *,
    persist_invalid_output: bool = False,
    preview_chars: int = 300,
) -> ModelAttempt:
    usage = usage or ModelUsage()
    head = tail = None
    if persist_invalid_output and text and not valid:
        head = text[:preview_chars]
        if len(text) > preview_chars:
            tail = text[-preview_chars:]
    return ModelAttempt(
        attempt=number,
        role="primary" if number == 1 else "repair",
        output_hash=canonical_hash(text),
        valid=valid,
        char_count=len(text),
        generation_input_tokens=usage.input_tokens,
        generation_output_tokens=usage.output_tokens,
        json_candidate_detected=valid or has_json_candidate(text),
        error_category=category,
        validation_error=error_text,
        output_preview=head,
        output_preview_tail=tail,
    )
