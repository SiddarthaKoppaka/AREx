"""Raw recent-turn window and artifact-referenced event views."""

from collections.abc import Callable

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.retrieval import ContextEvent
from arc_agi_3.trace.store import JsonlEventStore


def recent_turn_events(
    history: list[EventEnvelope], turn_limit: int, event_limit: int
) -> list[EventEnvelope]:
    if turn_limit <= 0 or not history:
        return []
    steps = sorted({event.step_id for event in history})[-turn_limit:]
    while len(steps) > 1:
        count = sum(event.step_id in steps for event in history)
        if count <= event_limit:
            break
        steps.pop(0)
    selected = set(steps)
    return [event for event in history if event.step_id in selected]


def event_view(
    events: JsonlEventStore,
    event: EventEnvelope,
    project: Callable[[EventEnvelope], ContextEvent],
    compact: bool,
) -> ContextEvent:
    visible = project(event)
    reference = events.artifacts.reference(event) if compact else None
    if reference is None:
        return visible
    return visible.model_copy(
        update={"payload": visible.payload | {"artifact_ref": reference}}
    )
