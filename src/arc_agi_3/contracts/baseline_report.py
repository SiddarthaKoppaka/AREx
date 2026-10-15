"""A machine-readable snapshot of one run: provenance plus metrics.

Wraps `EvaluationMetrics` rather than extending it, since that contract is
golden-pinned (`tests/golden/fake_run_signature.json`).
"""

from pydantic import JsonValue

from .base import Contract
from .evaluation import EvaluationMetrics


class ExocortexCoverage(Contract):
    """Distinguishes "no specialist backend was configured" from "one was
    configured but produced zero reports" — both are `0` in `EvaluationMetrics`
    alone, but only the latter is a real, meaningful zero."""

    configured: bool
    reports_generated: int | None = None
    reviews_generated: int | None = None
    backend_generations: int | None = None
    contradictions_raised: int | None = None


class BaselineReport(Contract):
    run_id: str
    experiment_id: str
    game_id: str
    seed: int
    config_hash: str
    last_event_hash: str
    event_count: int
    model: dict[str, JsonValue]
    environment: dict[str, JsonValue]
    ablations: dict[str, JsonValue]
    stop_reason: str | None
    context_ceiling_failures: int
    exocortex: ExocortexCoverage
    metrics: EvaluationMetrics
