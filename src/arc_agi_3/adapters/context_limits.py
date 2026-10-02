"""Resolve a backend's token counter and compaction thresholds, if it has them."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextLimits:
    counter: Callable[[str, dict[str, Any]], int]
    pressure: int
    target: int
    hard: int


def resolve_limits(backend: object) -> ContextLimits | None:
    counter = getattr(backend, "input_token_count", None)
    config = getattr(backend, "config", None)
    pressure = getattr(config, "compaction_pressure_start", None)
    if pressure is None:
        pressure = getattr(config, "soft_input_limit", None)
    target = getattr(config, "active_context_target", pressure)
    hard = getattr(config, "max_input_tokens", None)
    effective = getattr(backend, "effective_max_input_tokens", None)
    if isinstance(effective, int):
        hard = min(hard, effective) if isinstance(hard, int) else effective
    window = getattr(config, "max_context_tokens", None)
    output = getattr(config, "max_new_tokens", None)
    margin = getattr(config, "template_and_generation_margin", None)
    if isinstance(window, int) and isinstance(output, int) and isinstance(margin, int):
        hard = min(hard, window - output - margin) if isinstance(hard, int) else None
    if (
        not callable(counter)
        or not isinstance(pressure, int)
        or not isinstance(target, int)
        or not isinstance(hard, int)
    ):
        return None
    return ContextLimits(counter, pressure, target, hard)
