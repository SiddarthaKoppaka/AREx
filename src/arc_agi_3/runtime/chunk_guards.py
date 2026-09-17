"""Objective guards for an LM-authored execution chunk."""

from arc_agi_3.contracts.enums import (
    EnvironmentState,
    EventType,
    ExecutionStatus,
    Resource,
)
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ActionChunk
from arc_agi_3.contracts.observation import Observation

from .action import ActionOutcome
from .io import EpisodeIO


def invalid_reason(
    io: EpisodeIO, observation: Observation, required_hash: str | None, action_id: int
) -> str | None:
    if required_hash and required_hash != observation.observation_hash:
        return "stale_precondition"
    if action_id not in observation.available_actions:
        return "invalid_action"
    remaining = io.ledger.remaining().get(Resource.ACTIONS)
    if remaining is not None and remaining <= 0:
        return "action_budget_exhausted"
    return None


def post_step(
    chunk: ActionChunk, outcome: ActionOutcome
) -> tuple[str | None, ExecutionStatus]:
    if outcome.observation.state in {EnvironmentState.WIN, EnvironmentState.GAME_OVER}:
        return "terminal_state", ExecutionStatus.HARD_STOP
    if chunk.stop_on_mismatch and not outcome.verification.passed:
        return "prediction_mismatch", ExecutionStatus.HARD_STOP
    threshold = chunk.soft_interrupt_changed_cells
    if threshold is not None and outcome.verification.delta.changed_cells >= threshold:
        return "authorized_change_threshold", ExecutionStatus.SOFT_INTERRUPT
    return None, ExecutionStatus.COMPLETE


def checkpoint_if_needed(
    io: EpisodeIO, turn: int, chunk: ActionChunk, outcome: ActionOutcome
) -> None:
    if not chunk.checkpoint_after_each and turn % io.config.checkpoint_every == 0:
        io.checkpoint(turn, outcome.observation, outcome.verification_event.event_id)


def record_interrupt(
    io: EpisodeIO,
    turn: int,
    decision: EventEnvelope,
    started: EventEnvelope,
    index: int,
    status: ExecutionStatus,
    reason: str,
) -> None:
    io.append(
        EventType.INTERRUPT,
        "executor",
        turn,
        {"status": status, "reason": reason, "step_index": index},
        (decision.event_id, started.event_id),
    )
