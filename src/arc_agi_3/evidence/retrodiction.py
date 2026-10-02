"""Check a prediction against already-observed transitions (no simulation).

Extension point: an executable world model can later supply predicted
observations here; v1 compares only observable claims with recorded frames.
"""

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.retrodiction import (
    RetrodictionItem,
    RetrodictionRequest,
    RetrodictionResult,
)

from .history import transition_records
from .verdict import verify_outcome


def retrodict(
    events: list[EventEnvelope], request: RetrodictionRequest
) -> RetrodictionResult:
    records = transition_records(events)
    if request.action is not None:
        records = [item for item in records if item.action == request.action]
    buckets: dict[str, list[RetrodictionItem]] = {
        "matched": [],
        "mismatched": [],
        "unchecked": [],
    }
    selected = records[-request.limit :]
    for record in selected:
        pair = record.observations()
        status: str = "unchecked"
        mismatches: tuple[str, ...] = ()
        if pair is not None:
            result = verify_outcome(request.prediction, *pair)
            status, mismatches = result.status, result.mismatches
        buckets[status].append(
            RetrodictionItem(
                transition_event_id=record.transition_event.event_id,
                action_event_id=record.action_event.event_id,
                step_id=record.transition_event.step_id,
                mismatches=mismatches,
            )
        )
    return RetrodictionResult(
        compatible=tuple(buckets["matched"]),
        contradicting=tuple(buckets["mismatched"]),
        not_evaluable=tuple(buckets["unchecked"]),
        examined=len(selected),
    )
