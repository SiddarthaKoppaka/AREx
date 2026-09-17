"""Checkpoint restoration and immutable branch forking."""

from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.recovery import RecoveryRequest

from .io import EpisodeIO
from .recovery_state import RecoveryOutcome


def fork_checkpoint(
    io: EpisodeIO,
    turn: int,
    current: Observation,
    observed: EventEnvelope,
    requested: EventEnvelope,
    request: RecoveryRequest,
) -> RecoveryOutcome:
    checkpoint_id, new_branch = request.checkpoint_id, request.new_branch_id
    if checkpoint_id is None or new_branch is None:
        return failed(io, turn, current, observed, requested, "missing_target")
    if any(item.branch_id == new_branch for item in io.branches.records):
        return failed(io, turn, current, observed, requested, "duplicate_branch")
    checkpoint = io.checkpoints.load(checkpoint_id)
    if checkpoint.environment_state is None:
        return failed(io, turn, current, observed, requested, "restore_unsupported")
    workspace = CognitiveWorkspace(io.config.belief)
    workspace.restore(checkpoint.cognitive_state)
    restored = io.environment.restore(checkpoint.environment_state)
    expected_hash = checkpoint.observation.get("observation_hash")
    if expected_hash != restored.observation_hash:
        raise RuntimeError("environment restore did not reproduce checkpoint")
    frozen, child = io.branches.fork(new_branch, checkpoint_id)
    frozen_event = io.append(
        EventType.BRANCH_FROZEN,
        "recovery",
        turn,
        {"branch": frozen.model_dump(mode="json")},
        (requested.event_id,),
    )
    forked = io.append(
        EventType.BRANCH_FORKED,
        "recovery",
        turn,
        {"branch": child.model_dump(mode="json")},
        (requested.event_id, frozen_event.event_id),
    )
    io.events.branch_id = new_branch
    io.workspace = workspace
    restored_event = io.append(
        EventType.OBSERVATION,
        "environment",
        turn,
        restored.model_dump(mode="json"),
        (forked.event_id,),
    )
    io.append(
        EventType.RECOVERY,
        "recovery",
        turn,
        {"status": "complete"},
        (requested.event_id, forked.event_id),
    )
    return RecoveryOutcome(restored, restored_event)


def failed(
    io: EpisodeIO,
    turn: int,
    observation: Observation,
    observed: EventEnvelope,
    requested: EventEnvelope,
    reason: str,
) -> RecoveryOutcome:
    io.append(
        EventType.RECOVERY,
        "recovery",
        turn,
        {"status": "failed", "reason": reason},
        (requested.event_id,),
    )
    return RecoveryOutcome(observation, observed)
