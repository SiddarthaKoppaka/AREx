"""Separate action authorization from externally-owned environment execution."""

from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ExpectedOutcome
from arc_agi_3.contracts.observation import Action, Observation
from arc_agi_3.verification import verify_outcome

from .action_types import ActionOutcome, AuthorizedAction
from .io import EpisodeIO


def authorize_action(
    io: EpisodeIO,
    turn: int,
    before: Observation,
    decision: EventEnvelope,
    action: Action,
    expected: ExpectedOutcome | None,
    *,
    checkpoint: bool,
    checkpoint_suffix: int | None = None,
    extra_causes: tuple[str, ...] = (),
) -> AuthorizedAction:
    if action.action_id not in before.available_actions:
        raise ValueError("LM-selected action is not currently authorized")
    io.spend(Resource.ACTIONS, 1, turn, (decision.event_id,))
    acted = io.append(
        EventType.ACTION,
        "executor",
        turn,
        {
            "action": action.model_dump(mode="json"),
            "before_hash": before.observation_hash,
        },
        (decision.event_id, *extra_causes),
    )
    return AuthorizedAction(
        turn, before, decision, action, expected, acted, checkpoint, checkpoint_suffix
    )


def record_action_result(
    io: EpisodeIO, pending: AuthorizedAction, after: Observation
) -> ActionOutcome:
    observed = io.append(
        EventType.OBSERVATION,
        "environment",
        pending.turn,
        after.model_dump(mode="json"),
        (pending.action_event.event_id,),
    )
    result = verify_outcome(pending.expected, pending.before, after)
    transition = io.append(
        EventType.TRANSITION,
        "verifier",
        pending.turn,
        result.delta.model_dump(mode="json"),
        (pending.action_event.event_id, observed.event_id),
    )
    verified = io.append(
        EventType.VERIFICATION,
        "verifier",
        pending.turn,
        result.model_dump(mode="json"),
        (pending.decision.event_id, transition.event_id),
    )
    if pending.checkpoint:
        io.checkpoint(pending.turn, after, verified.event_id, pending.checkpoint_suffix)
    return ActionOutcome(after, observed, result, verified)
