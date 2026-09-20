"""Objective model-resource accounting."""

from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.contracts.enums import EventType, Resource

from .io import EpisodeIO

STOP_REASONS = {
    Resource.MODEL_CALLS: "model_call_budget_exhausted",
    Resource.ACTIONS: "action_budget_exhausted",
    Resource.INPUT_TOKENS: "input_token_budget_exhausted",
    Resource.OUTPUT_TOKENS: "output_token_budget_exhausted",
    Resource.WALL_TIME_MS: "wall_time_budget_exhausted",
}


def exhausted_reason(io: EpisodeIO, *resources: Resource) -> str | None:
    remaining = io.ledger.remaining()
    for resource in resources:
        if resource in remaining and remaining[resource] <= 0:
            return STOP_REASONS[resource]
    return None


def record_model_usage(io: EpisodeIO, usage: ModelUsage, step: int, cause: str) -> None:
    for resource, amount in (
        (Resource.INPUT_TOKENS, usage.input_tokens),
        (Resource.OUTPUT_TOKENS, usage.output_tokens),
        (Resource.WALL_TIME_MS, usage.latency_ms),
    ):
        if amount:
            io.ledger.record_actual(resource, amount)
            io.append(
                EventType.BUDGET,
                "budget",
                step,
                {"resource": resource, "amount": amount},
                (cause,),
            )
