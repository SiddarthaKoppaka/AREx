"""Reconstruct a small exact window around the first objective divergence."""

from arc_agi_3.context import compact_context_event, context_event
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.recovery import RecoveryEvidence


def reconstruct_recovery(
    events: list[EventEnvelope], radius: int = 2, *, compact: bool = True
) -> RecoveryEvidence | None:
    from arc_agi_3.runtime.divergence import find_first_divergence

    divergence = find_first_divergence(events)
    if divergence is None:
        return None
    start = max(0, divergence.sequence - radius)
    end = divergence.sequence + radius + 1
    project = compact_context_event if compact else context_event
    local = tuple(project(event) for event in events[start:end])
    checkpoints = tuple(
        str(event.payload["checkpoint_id"])
        for event in events[: divergence.sequence + 1]
        if event.event_type is EventType.CHECKPOINT
        and isinstance(event.payload.get("checkpoint_id"), str)
    )
    return RecoveryEvidence(
        divergence_event_id=divergence.event_id,
        divergence_sequence=divergence.sequence,
        reason=divergence.reason,
        local_events=local,
        checkpoint_ids=checkpoints,
    )
