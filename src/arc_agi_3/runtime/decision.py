"""Persist one LM response before any requested operation is applied."""

from dataclasses import dataclass

from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.contracts.decision import ModelResponse
from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation

from .io import EpisodeIO
from .usage import record_model_usage


@dataclass(frozen=True)
class RecordedDecision:
    response: ModelResponse
    event: EventEnvelope


def request_decision(
    io: EpisodeIO,
    model: ModelAdapter,
    turn: int,
    observation: Observation,
    observed: EventEnvelope,
) -> RecordedDecision:
    budget = io.spend(Resource.MODEL_CALLS, 1, turn)
    response = model.decide(io.context(turn, observation))
    event = io.append(
        EventType.MODEL_DECISION,
        "model",
        turn,
        {
            "decision": response.decision.model_dump(mode="json"),
            "usage": response.usage.model_dump(mode="json"),
            "attempts": [item.model_dump(mode="json") for item in response.attempts],
        },
        (budget.event_id, observed.event_id),
    )
    record_model_usage(io, response, turn, event.event_id)
    return RecordedDecision(response, event)
