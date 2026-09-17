"""Execute only the recovery operation explicitly selected by the LM."""

from arc_agi_3.contracts.enums import EventType, RecoveryOperation
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.recovery import RecoveryRequest

from .io import EpisodeIO
from .recovery_fork import fork_checkpoint
from .recovery_state import RecoveryOutcome


def execute_recovery(
    io: EpisodeIO,
    turn: int,
    observation: Observation,
    observed: EventEnvelope,
    decision: EventEnvelope,
    request: RecoveryRequest,
) -> RecoveryOutcome:
    requested = io.append(
        EventType.RECOVERY,
        "recovery",
        turn,
        {"status": "requested", "request": request.model_dump(mode="json")},
        (decision.event_id,),
    )
    if request.operation is RecoveryOperation.ABANDON_BRANCH:
        branch = io.branches.abandon()
        io.append(
            EventType.BRANCH_FROZEN,
            "recovery",
            turn,
            {"branch": branch.model_dump(mode="json")},
            (requested.event_id,),
        )
        return RecoveryOutcome(observation, observed, "agent_abandon")
    return fork_checkpoint(io, turn, observation, observed, requested, request)
