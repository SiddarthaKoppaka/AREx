"""Locate the first objective prediction or authorization divergence."""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import DivergenceRecord


def find_first_divergence(events: list[EventEnvelope]) -> DivergenceRecord | None:
    for event in events:
        if event.event_type is EventType.VERIFICATION:
            checked, passed = event.payload.get("checked"), event.payload.get("passed")
            if checked is True and passed is False:
                mismatches = event.payload.get("mismatches")
                labels = mismatches if isinstance(mismatches, list) else []
                return DivergenceRecord(
                    event_id=event.event_id,
                    sequence=event.sequence,
                    step_id=event.step_id,
                    reason="prediction_mismatch:"
                    + ",".join(str(item) for item in labels),
                    evidence_refs=event.causal_refs,
                )
        if event.event_type is EventType.INTERRUPT:
            reason = event.payload.get("reason")
            if isinstance(reason, str):
                return DivergenceRecord(
                    event_id=event.event_id,
                    sequence=event.sequence,
                    step_id=event.step_id,
                    reason=reason,
                    evidence_refs=event.causal_refs,
                )
    return None
