"""Stepwise execution beneath a bounded LM-authored chunk authorization."""

from arc_agi_3.contracts.enums import EventType, ExecutionStatus
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ActionChunk
from arc_agi_3.contracts.observation import Observation

from .action import ActionOutcome, execute_action
from .chunk_guards import (
    checkpoint_if_needed,
    invalid_reason,
    post_step,
    record_interrupt,
)
from .io import EpisodeIO


def execute_chunk(
    io: EpisodeIO,
    turn: int,
    observation: Observation,
    decision: EventEnvelope,
    chunk: ActionChunk,
) -> ActionOutcome | None:
    started = io.append(
        EventType.CHUNK_STARTED,
        "executor",
        turn,
        {"chunk": chunk.model_dump(mode="json")},
        (decision.event_id,),
    )
    current = observation
    last: ActionOutcome | None = None
    for index, step in enumerate(chunk.steps):
        reason = invalid_reason(
            io, current, step.required_before_hash, step.action.action_id
        )
        if reason:
            if last is not None:
                checkpoint_if_needed(io, turn, chunk, last)
            record_interrupt(
                io, turn, decision, started, index, ExecutionStatus.HARD_STOP, reason
            )
            return last
        last = execute_action(
            io,
            turn,
            current,
            decision,
            step.action,
            (
                step.expected_outcome
                if io.config.ablations.prediction_verification
                else None
            ),
            checkpoint=chunk.checkpoint_after_each,
            checkpoint_suffix=index + 1 if chunk.checkpoint_after_each else None,
            extra_causes=(started.event_id,),
        )
        current = last.observation
        reason, status = post_step(chunk, last)
        if reason:
            checkpoint_if_needed(io, turn, chunk, last)
            record_interrupt(io, turn, decision, started, index, status, reason)
            return last
    if last is not None:
        checkpoint_if_needed(io, turn, chunk, last)
    io.append(
        EventType.CHUNK_FINISHED,
        "executor",
        turn,
        {"chunk_id": chunk.chunk_id, "status": ExecutionStatus.COMPLETE},
        (decision.event_id, started.event_id),
    )
    return last
