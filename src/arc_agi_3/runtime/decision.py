"""Persist one LM response before any requested operation is applied."""

from dataclasses import dataclass

from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.adapters.structured_output import StructuredOutputError
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
    overrun_reason: str | None = None


def request_decision(
    io: EpisodeIO,
    model: ModelAdapter,
    turn: int,
    observation: Observation,
    observed: EventEnvelope,
) -> RecordedDecision:
    budget = io.spend(Resource.MODEL_CALLS, 1, turn)
    try:
        response = model.decide(io.context(turn, observation))
    except StructuredOutputError as error:
        record_model_usage(io, error.usage, turn, budget.event_id)
        raise
    event = io.append(
        EventType.MODEL_DECISION,
        "model",
        turn,
        {
            "decision": response.decision.model_dump(mode="json"),
            "usage": response.usage.model_dump(mode="json"),
            "attempts": [
                item.model_dump(mode="json", exclude_none=True)
                for item in response.attempts
            ],
        },
        (budget.event_id, observed.event_id),
    )
    record_model_usage(io, response.usage, turn, event.event_id)
    from .usage import exhausted_reason

    overrun = None
    remaining = io.ledger.remaining()
    for resource in (
        Resource.INPUT_TOKENS,
        Resource.OUTPUT_TOKENS,
        Resource.WALL_TIME_MS,
    ):
        if resource in remaining and remaining[resource] < 0:
            overrun = exhausted_reason(io, resource)
            break
    return RecordedDecision(response, event, overrun)
