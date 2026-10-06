"""A specialist's own prior reports, read from the immutable trace.

Continuity across turns (a Critic remembering it was skeptical) is exact
retrieval, not a new mutable store — the same pattern as
`evidence/action_cues.py` filtering transitions by action.
"""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.specialist_report import SpecialistReport


def recent_reports(
    history: list[EventEnvelope], specialist_id: str, limit: int = 2
) -> tuple[SpecialistReport, ...]:
    own = [
        report
        for event in history
        if event.event_type is EventType.SPECIALIST_REPORT
        and isinstance((report := event.payload.get("report")), dict)
        and report.get("specialist_id") == specialist_id
    ]
    return tuple(SpecialistReport.model_validate(payload) for payload in own[-limit:])
