"""Classroom-round counters derived mechanically from immutable events."""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def _list_length(event: EventEnvelope, key: str) -> int:
    value = event.payload.get(key)
    return len(value) if isinstance(value, list) else 0


def classroom_counts(events: list[EventEnvelope]) -> dict[str, int]:
    reports = [e for e in events if e.event_type is EventType.STUDENT_REPORT]
    reviews = [e for e in events if e.event_type is EventType.PEER_REVIEW]
    rounds = [e for e in events if e.event_type is EventType.CLASSROOM_SYNTHESIS]
    flagged = sum(_list_length(e, "flagged_contradictions") for e in rounds)
    attempts = sum(_list_length(e, "attempts") for e in (*reports, *reviews))
    return {
        "student_reports_generated": len(reports),
        "peer_reviews_generated": len(reviews),
        "classroom_rounds": len(rounds),
        "classroom_backend_generations": attempts,
        "flagged_contradictions_raised": flagged,
    }
