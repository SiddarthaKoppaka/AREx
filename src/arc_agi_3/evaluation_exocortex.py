"""ExoCortex consultation counters derived mechanically from immutable events."""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def _list_length(event: EventEnvelope, key: str) -> int:
    value = event.payload.get(key)
    return len(value) if isinstance(value, list) else 0


def _critique_count(event: EventEnvelope) -> int:
    review = event.payload.get("review")
    if not isinstance(review, dict):
        return 0
    critiques = review.get("critiques")
    return len(critiques) if isinstance(critiques, list) else 0


def _structured_output_failures(events: list[EventEnvelope]) -> int:
    """A specialist tool call failed with `StructuredOutputError` (caught
    in `runtime/tools.py` as an ordinary failed `ToolResult`, never a
    `FAILURE` event) - distinct from `structured_output_failures`
    (`evaluation_generation.py`), which only counts the primary model."""
    results = [e for e in events if e.event_type is EventType.TOOL_RESULT]
    return sum(
        isinstance((result := e.payload.get("result")), dict)
        and result.get("status") == "failed"
        and isinstance((output := result.get("output")), dict)
        and output.get("error_type") == "StructuredOutputError"
        for e in results
    )


def exocortex_counts(events: list[EventEnvelope]) -> dict[str, int]:
    reports = [e for e in events if e.event_type is EventType.SPECIALIST_REPORT]
    reviews = [e for e in events if e.event_type is EventType.SPECIALIST_REVIEW]
    attempts = sum(_list_length(e, "attempts") for e in (*reports, *reviews))
    flagged = sum(_critique_count(e) for e in reviews)
    return {
        "specialist_reports_generated": len(reports),
        "specialist_reviews_generated": len(reviews),
        "exocortex_backend_generations": attempts,
        "flagged_contradictions_raised": flagged,
        "specialist_structured_output_failures": _structured_output_failures(events),
    }
