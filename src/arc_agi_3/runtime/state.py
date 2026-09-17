"""Runtime return values."""

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.evaluation import EvaluationMetrics


class RunResult(Contract):
    run_id: str
    stop_reason: str
    trace_path: str
    metrics: EvaluationMetrics
