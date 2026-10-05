"""A Student's own prior reports, read from the immutable trace.

Continuity across turns (a Skeptic remembering it was skeptical) is exact
retrieval, not a new mutable store — the same pattern as
`evidence/action_cues.py` filtering transitions by action.
"""

from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def recent_reports(
    history: list[EventEnvelope], student_id: str, limit: int = 2
) -> tuple[StudentReport, ...]:
    own = [
        event.payload
        for event in history
        if event.event_type is EventType.STUDENT_REPORT
        and event.payload.get("student_id") == student_id
    ]
    return tuple(StudentReport.model_validate(payload) for payload in own[-limit:])
