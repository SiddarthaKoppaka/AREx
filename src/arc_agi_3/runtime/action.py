"""Execute one already authorized action and persist objective evidence."""

from dataclasses import dataclass

from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ExpectedOutcome
from arc_agi_3.contracts.observation import Action, Observation
from arc_agi_3.contracts.verification import VerificationResult
from arc_agi_3.verification import verify_outcome

from .io import EpisodeIO


@dataclass(frozen=True)
class ActionOutcome:
    observation: Observation
    observed_event: EventEnvelope
    verification: VerificationResult
    verification_event: EventEnvelope


def execute_action(
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
) -> ActionOutcome:
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
    after = io.environment.step(action)
    observed = io.append(
        EventType.OBSERVATION,
        "environment",
        turn,
        after.model_dump(mode="json"),
        (acted.event_id,),
    )
    result = verify_outcome(expected, before, after)
    transition = io.append(
        EventType.TRANSITION,
        "verifier",
        turn,
        result.delta.model_dump(mode="json"),
        (acted.event_id, observed.event_id),
    )
    verified = io.append(
        EventType.VERIFICATION,
        "verifier",
        turn,
        result.model_dump(mode="json"),
        (decision.event_id, transition.event_id),
    )
    if checkpoint:
        io.checkpoint(turn, after, verified.event_id, checkpoint_suffix)
    return ActionOutcome(after, observed, result, verified)
