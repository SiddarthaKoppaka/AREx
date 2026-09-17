"""One LM-controlled turn with deterministic execution and verification."""

from dataclasses import dataclass

from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.contracts.enums import DecisionMode, EventType, Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation

from .action import execute_action
from .chunks import execute_chunk
from .cognition import apply_cognitive_decision
from .io import EpisodeIO
from .recovery import execute_recovery
from .usage import record_model_usage


@dataclass(frozen=True)
class TurnOutcome:
    observation: Observation
    observed_event: EventEnvelope
    stop_reason: str | None = None


def execute_turn(
    io: EpisodeIO,
    model: ModelAdapter,
    turn: int,
    observation: Observation,
    observed_event: EventEnvelope,
) -> TurnOutcome:
    budget = io.spend(Resource.MODEL_CALLS, 1, turn)
    response = model.decide(io.context(turn, observation))
    decision = io.append(
        EventType.MODEL_DECISION,
        "model",
        turn,
        {
            "decision": response.decision.model_dump(mode="json"),
            "usage": response.usage.model_dump(mode="json"),
            "attempts": [item.model_dump(mode="json") for item in response.attempts],
        },
        (budget.event_id, observed_event.event_id),
    )
    record_model_usage(io, response, turn, decision.event_id)
    if response.decision.mode is DecisionMode.RECOVER:
        request = response.decision.recovery
        if request is None:
            raise ValueError("recover decision lacks request")
        recovered = execute_recovery(
            io, turn, observation, observed_event, decision, request
        )
        if recovered.stop_reason:
            return TurnOutcome(
                recovered.observation, recovered.observed_event, recovered.stop_reason
            )
        apply_cognitive_decision(io, decision, response.decision, turn)
        return TurnOutcome(recovered.observation, recovered.observed_event)
    apply_cognitive_decision(io, decision, response.decision, turn)
    if response.decision.mode is DecisionMode.STOP:
        return TurnOutcome(observation, observed_event, "agent_stop")
    if response.decision.mode is not DecisionMode.EXECUTE:
        return TurnOutcome(observation, observed_event)
    chunk = response.decision.action_chunk
    if chunk is not None:
        outcome = execute_chunk(io, turn, observation, decision, chunk)
        if outcome is None:
            return TurnOutcome(observation, observed_event)
        return TurnOutcome(outcome.observation, outcome.observed_event)
    action = response.decision.action
    if action is None:
        raise ValueError("execute decision has no authorization")
    outcome = execute_action(
        io,
        turn,
        observation,
        decision,
        action,
        (
            response.decision.expected_outcome
            if io.config.ablations.prediction_verification
            else None
        ),
        checkpoint=turn % io.config.checkpoint_every == 0,
    )
    return TurnOutcome(outcome.observation, outcome.observed_event)
