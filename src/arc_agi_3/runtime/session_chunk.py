"""Carry one LM-authorized chunk across environment-owned callbacks."""

from arc_agi_3.contracts.enums import EventType, ExecutionStatus
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ActionChunk
from arc_agi_3.contracts.observation import Observation

from .action_phases import authorize_action
from .action_types import ActionOutcome, AuthorizedAction
from .chunk_guards import (
    checkpoint_if_needed,
    invalid_reason,
    post_step,
    record_interrupt,
)
from .io import EpisodeIO
from .session_types import ActiveChunk


def begin_chunk(
    io: EpisodeIO, turn: int, decision: EventEnvelope, chunk: ActionChunk
) -> ActiveChunk:
    started = io.append(
        EventType.CHUNK_STARTED,
        "executor",
        turn,
        {"chunk": chunk.model_dump(mode="json")},
        (decision.event_id,),
    )
    return ActiveChunk(chunk, decision, started)


def next_chunk_action(
    io: EpisodeIO, turn: int, observation: Observation, active: ActiveChunk
) -> AuthorizedAction | None:
    step = active.chunk.steps[active.index]
    reason = invalid_reason(
        io, observation, step.required_before_hash, step.action.action_id
    )
    if reason:
        if active.last is not None:
            checkpoint_if_needed(io, turn, active.chunk, active.last)
        record_interrupt(
            io,
            turn,
            active.decision,
            active.started,
            active.index,
            ExecutionStatus.HARD_STOP,
            reason,
        )
        return None
    expected = (
        step.expected_outcome if io.config.ablations.prediction_verification else None
    )
    return authorize_action(
        io,
        turn,
        observation,
        active.decision,
        step.action,
        expected,
        checkpoint=active.chunk.checkpoint_after_each,
        checkpoint_suffix=active.index + 1
        if active.chunk.checkpoint_after_each
        else None,
        extra_causes=(active.started.event_id,),
    )


def accept_chunk_result(
    io: EpisodeIO, turn: int, active: ActiveChunk, outcome: ActionOutcome
) -> bool:
    active.last = outcome
    reason, status = post_step(active.chunk, outcome)
    if reason:
        checkpoint_if_needed(io, turn, active.chunk, outcome)
        record_interrupt(
            io, turn, active.decision, active.started, active.index, status, reason
        )
        return False
    active.index += 1
    if active.index < len(active.chunk.steps):
        return True
    checkpoint_if_needed(io, turn, active.chunk, outcome)
    io.append(
        EventType.CHUNK_FINISHED,
        "executor",
        turn,
        {"chunk_id": active.chunk.chunk_id, "status": ExecutionStatus.COMPLETE},
        (active.decision.event_id, active.started.event_id),
    )
    return False
