"""Hypotheses each executed action was declared to test (by prediction or frame)."""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope

from .history import TransitionRecord


def experiment_targets(events: list[EventEnvelope]) -> dict[str, tuple[str, ...]]:
    """Hypothesis IDs each experiment-framed action event claimed to test."""
    targets: dict[str, tuple[str, ...]] = {}
    for event in events:
        experiment = event.payload.get("experiment")
        if event.event_type is EventType.EXPERIMENT and isinstance(experiment, dict):
            ids = experiment.get("hypothesis_ids")
            action_id = event.payload.get("action_event_id")
            if isinstance(ids, list) and isinstance(action_id, str):
                targets[action_id] = tuple(str(item) for item in ids)
    return targets


def targeted(record: TransitionRecord, targets: dict[str, tuple[str, ...]]) -> set[str]:
    return {
        *record.tested_hypotheses,
        *targets.get(record.action_event.event_id, ()),
    }
