"""Exact changed-cell views derived from observation frames."""

from pydantic import TypeAdapter

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Frame

_FRAME = TypeAdapter(Frame)


def transition_delta(
    event: EventEnvelope, history: list[EventEnvelope]
) -> dict[str, object]:
    observations = [
        item
        for item in history
        if item.event_type is EventType.OBSERVATION
        and item.branch_id == event.branch_id
    ]
    if event.event_type is EventType.TRANSITION:
        before_hash = event.payload.get("before_hash")
        after_hash = event.payload.get("after_hash")
        causal = set(event.causal_refs)
        after = next((item for item in observations if item.event_id in causal), None)
        if after is None or after.payload.get("observation_hash") != after_hash:
            raise ValueError("transition has no matching causal observation")
        earlier = [
            item
            for item in observations
            if item.sequence < after.sequence
            and item.payload.get("observation_hash") == before_hash
        ]
        before = earlier[-1] if earlier else None
    elif event.event_type is EventType.OBSERVATION:
        prior = [item for item in observations if item.sequence < event.sequence]
        before, after = (prior[-1] if prior else None), event
    else:
        raise ValueError("transition_delta requires an observation or transition")
    if before is None or after is None:
        raise ValueError("matching before and after observations are unavailable")
    left_data, right_data = before.payload.get("frame"), after.payload.get("frame")
    if not isinstance(left_data, list) or not isinstance(right_data, list):
        raise ValueError("transition observations have no frames")
    left = _FRAME.validate_python(left_data)
    right = _FRAME.validate_python(right_data)
    old = {
        (z, y, x): value
        for z, layer in enumerate(left)
        for y, row in enumerate(layer)
        for x, value in enumerate(row)
    }
    new = {
        (z, y, x): value
        for z, layer in enumerate(right)
        for y, row in enumerate(layer)
        for x, value in enumerate(row)
    }
    changes = [
        [*key, old.get(key), new.get(key)]
        for key in sorted(old.keys() | new.keys())
        if old.get(key) != new.get(key)
    ]
    return {
        "before_event_id": before.event_id,
        "after_event_id": after.event_id,
        "changed_cell_count": len(changes),
        "changes": changes,
    }
