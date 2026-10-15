"""Report-level fields derived mechanically from immutable events.

Separate from `evaluation_counts.py`'s `EvaluationMetrics` counters because
these two are not part of that golden-pinned contract.
"""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def stop_reason(events: list[EventEnvelope]) -> str | None:
    for event in reversed(events):
        if event.event_type is EventType.RUN_FINISHED:
            reason = event.payload.get("reason")
            return reason if isinstance(reason, str) else None
    return None


def context_ceiling_failures(events: list[EventEnvelope]) -> int:
    return sum(
        event.event_type is EventType.FAILURE
        and event.payload.get("error_type") == "InputTokenLimitError"
        for event in events
    )
