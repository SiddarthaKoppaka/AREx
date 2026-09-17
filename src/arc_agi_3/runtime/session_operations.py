"""Small operations used by the callback-driven session controller."""

from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.recovery import RecoveryRequest

from .action_phases import authorize_action
from .action_types import AuthorizedAction
from .io import EpisodeIO
from .recovery import execute_recovery
from .recovery_state import RecoveryOutcome


def authorize_direct(
    io: EpisodeIO,
    turn: int,
    observation: Observation,
    decision: EventEnvelope,
    choice: CognitiveDecision,
) -> AuthorizedAction:
    if choice.action is None:
        raise ValueError("execute decision has no authorization")
    expected = (
        choice.expected_outcome if io.config.ablations.prediction_verification else None
    )
    return authorize_action(
        io,
        turn,
        observation,
        decision,
        choice.action,
        expected,
        checkpoint=turn % io.config.checkpoint_every == 0,
    )


def recover_session(
    io: EpisodeIO,
    turn: int,
    observation: Observation,
    observed: EventEnvelope,
    decision: EventEnvelope,
    request: RecoveryRequest | None,
) -> RecoveryOutcome:
    if request is None:
        raise ValueError("recover decision lacks request")
    return execute_recovery(io, turn, observation, observed, decision, request)
