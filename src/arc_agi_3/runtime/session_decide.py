"""Advance LM cognition until it authorizes an action or ends the session."""

from dataclasses import dataclass

from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation

from .action_types import AuthorizedAction
from .cognition import apply_cognitive_decision
from .decision import request_decision
from .io import EpisodeIO
from .session_chunk import begin_chunk, next_chunk_action
from .session_operations import authorize_direct, recover_session
from .session_types import ActiveChunk


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
        turn += 1
        recorded = request_decision(io, model, turn, observation, observed)
        choice, decision = recorded.response.decision, recorded.event
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
        if choice.action_chunk is None:
            pending = authorize_direct(io, turn, observation, decision, choice)
            return DecisionOutcome(turn, observation, observed, pending=pending)
        chunk = begin_chunk(io, turn, decision, choice.action_chunk)
        chunk_pending = next_chunk_action(io, turn, observation, chunk)
        if chunk_pending is not None:
            return DecisionOutcome(
                turn, observation, observed, pending=chunk_pending, chunk=chunk
            )
    return DecisionOutcome(turn, observation, observed, stop_reason="max_turns")
