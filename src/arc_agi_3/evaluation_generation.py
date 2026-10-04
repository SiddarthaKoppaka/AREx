"""Backend-generation counters derived mechanically from decision/failure events.

`model_calls` already counts logical decisions; these counters additionally
expose the real compute cost (every backend generation attempt), so a
runaway repair loop is visible even when the logical decision count stays low.
"""

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def _attempts(events: list[EventEnvelope]) -> list[dict[str, JsonValue]]:
    found: list[dict[str, JsonValue]] = []
    for event in events:
        if event.event_type is EventType.MODEL_DECISION:
            source = event.payload.get("attempts")
        elif event.event_type is EventType.FAILURE:
            details = event.payload.get("details")
            source = details.get("attempts") if isinstance(details, dict) else None
        else:
            continue
        if isinstance(source, list):
            found.extend(item for item in source if isinstance(item, dict))
    return found


def _logical_decisions(events: list[EventEnvelope]) -> int:
    return sum(
        event.event_type in {EventType.MODEL_DECISION, EventType.FAILURE}
        for event in events
    )


def generation_counts(events: list[EventEnvelope]) -> dict[str, int | float]:
    attempts = _attempts(events)
    primary = sum(item.get("role", "primary") == "primary" for item in attempts)
    repair = sum(item.get("role") == "repair" for item in attempts)
    failed = sum(item.get("valid") is False for item in attempts)
    backend = primary + repair
    decisions = _logical_decisions(events)
    structured_failures = sum(
        event.event_type is EventType.FAILURE
        and event.payload.get("error_type") == "StructuredOutputError"
        for event in events
    )
    repair_tokens = sum(
        value
        for item in attempts
        if item.get("role") == "repair"
        and isinstance((value := item.get("generation_input_tokens")), int)
    )
    return {
        "primary_generations": primary,
        "repair_generations": repair,
        "backend_generations": backend,
        "failed_generations": failed,
        "structured_output_failures": structured_failures,
        "repair_prompt_tokens": repair_tokens,
        "avg_backend_generations_per_decision": round(backend / decisions, 3)
        if decisions
        else 0.0,
    }
