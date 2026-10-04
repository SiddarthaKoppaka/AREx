"""Trace experiment framing, plan adherence, and repeat cues at authorization.

Recorded evidence only: a repeat never blocks an action, and a deviation from
the planned `next_test` is allowed and merely made visible.
"""

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Action, Observation
from arc_agi_3.contracts.prediction import ExpectedOutcome, Experiment
from arc_agi_3.evidence import detect_repeat, experiment_targets, transition_records

from .io import EpisodeIO


def plan_fields(io: EpisodeIO, action: Action) -> dict[str, JsonValue]:
    planned = io.workspace.scratchpad.current.next_test
    if planned is None:
        return {}
    return {
        "planned_next_test": planned.model_dump(mode="json"),
        "matches_planned_next_test": planned.action_id == action.action_id,
    }


def record_experiment(
    io: EpisodeIO,
    turn: int,
    decision: EventEnvelope,
    acted: EventEnvelope,
    before: Observation,
    action: Action,
    expected: ExpectedOutcome | None,
    experiment: Experiment | None,
) -> EventEnvelope | None:
    history = io.events.read()
    cue = detect_repeat(
        transition_records(history),
        experiment_targets(history),
        before_hash=before.observation_hash,
        action=action,
        expected=expected,
        experiment=experiment,
    )
    if experiment is None and not cue.equivalent_prior:
        return None
    payload: dict[str, JsonValue] = {
        "action_event_id": acted.event_id,
        "action": action.model_dump(mode="json"),
        "experiment": experiment.model_dump(mode="json") if experiment else None,
        "prediction_id": expected.prediction_id if expected else None,
        "has_checkable_prediction": bool(expected and expected.checkable),
        "repeat": cue.model_dump(mode="json"),
        "unjustified_repeat": cue.unjustified,
    }
    return io.append(
        EventType.EXPERIMENT,
        "verifier",
        turn,
        payload,
        (decision.event_id, acted.event_id),
    )
