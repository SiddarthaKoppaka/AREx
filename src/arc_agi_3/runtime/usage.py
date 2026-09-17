"""Objective model-resource accounting."""

from arc_agi_3.contracts.decision import ModelResponse
from arc_agi_3.contracts.enums import Resource

from .io import EpisodeIO


def record_model_usage(
    io: EpisodeIO, response: ModelResponse, step: int, cause: str
) -> None:
    usage = response.usage
    for resource, amount in (
        (Resource.INPUT_TOKENS, usage.input_tokens),
        (Resource.OUTPUT_TOKENS, usage.output_tokens),
        (Resource.WALL_TIME_MS, usage.latency_ms),
    ):
        if amount:
            io.spend(resource, amount, step, (cause,))
