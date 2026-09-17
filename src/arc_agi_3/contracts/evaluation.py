"""Deterministic run evaluation records."""

from pydantic import Field

from .base import Contract


class EvaluationMetrics(Contract):
    succeeded: bool
    levels_completed: int = Field(ge=0)
    environment_actions: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    verification_failures: int = Field(ge=0)
    failures: int = Field(ge=0)
    checkpoints: int = Field(ge=0)
    search_node_expansions: int = Field(default=0, ge=0)
    simulations: int = Field(default=0, ge=0)
    wall_time_ms: int = Field(default=0, ge=0)
    prediction_mismatches: int = Field(default=0, ge=0)
    soft_interrupts: int = Field(default=0, ge=0)
    hard_stops: int = Field(default=0, ge=0)
    recovery_attempts: int = Field(default=0, ge=0)
    recovery_failures: int = Field(default=0, ge=0)
    branch_forks: int = Field(default=0, ge=0)
    repeated_actions: int = Field(default=0, ge=0)
    world_model_versions: int = Field(default=0, ge=0)
    rhae_level_scores: tuple[float, ...] = ()
    rhae_environment_score: float | None = None
