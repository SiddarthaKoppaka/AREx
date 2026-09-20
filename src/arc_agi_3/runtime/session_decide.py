"""Advance LM cognition until it authorizes an action or ends the session."""

from dataclasses import dataclass

from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.contracts.enums import DecisionMode, Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation

from .action_types import AuthorizedAction
from .cognition import apply_cognitive_decision
from .decision import request_decision
from .io import EpisodeIO
from .session_chunk import begin_chunk, next_chunk_action
from .session_operations import authorize_direct, recover_session
from .session_types import ActiveChunk
from .usage import exhausted_reason


@dataclass(frozen=True)
class DecisionOutcome:
    turn: int
    observation: Observation
    observed: EventEnvelope
    pending: AuthorizedAction | None = None
    chunk: ActiveChunk | None = None
    stop_reason: str | None = None


def decide_until_action(
    io: EpisodeIO,
    model: ModelAdapter,
    observation: Observation,
    observed: EventEnvelope,
    turn: int,
) -> DecisionOutcome:
    while turn < io.config.max_turns:
        exhausted = exhausted_reason(
            io,
            Resource.MODEL_CALLS,
            Resource.ACTIONS,
            Resource.INPUT_TOKENS,
            Resource.OUTPUT_TOKENS,
            Resource.WALL_TIME_MS,
        )
        if exhausted:
            return DecisionOutcome(turn, observation, observed, stop_reason=exhausted)
        turn += 1
        recorded = request_decision(io, model, turn, observation, observed)
        choice, decision = recorded.response.decision, recorded.event
        if recorded.overrun_reason:
            return DecisionOutcome(
                turn, observation, observed, stop_reason=recorded.overrun_reason
            )
        if choice.mode is DecisionMode.RECOVER:
            recovered = recover_session(
                io, turn, observation, observed, decision, choice.recovery
            )
            observation, observed = recovered.observation, recovered.observed_event
            if recovered.stop_reason:
                return DecisionOutcome(
                    turn, observation, observed, stop_reason=recovered.stop_reason
                )
            continue
        apply_cognitive_decision(io, decision, choice, turn)
        if choice.mode is DecisionMode.STOP:
            return DecisionOutcome(
                turn, observation, observed, stop_reason="agent_stop"
            )
        if choice.mode is not DecisionMode.EXECUTE:
            continue
        exhausted = exhausted_reason(io, Resource.ACTIONS)
        if exhausted:
            return DecisionOutcome(turn, observation, observed, stop_reason=exhausted)
        if choice.action_chunk is None:
            pending = authorize_direct(io, turn, observation, decision, choice)
            return DecisionOutcome(turn, observation, observed, pending=pending)
        chunk = begin_chunk(io, turn, decision, choice.action_chunk)
        chunk_pending = next_chunk_action(io, turn, observation, chunk)
        if chunk_pending is not None:
            return DecisionOutcome(
                turn, observation, observed, pending=chunk_pending, chunk=chunk
            )
        exhausted = exhausted_reason(io, Resource.ACTIONS)
        if exhausted:
            return DecisionOutcome(turn, observation, observed, stop_reason=exhausted)
    return DecisionOutcome(turn, observation, observed, stop_reason="max_turns")
