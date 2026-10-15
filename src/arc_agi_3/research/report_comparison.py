"""Control-vs-candidate deltas over two `BaselineReport`s, missing-preserving."""

from arc_agi_3.contracts.baseline_report import BaselineReport
from arc_agi_3.contracts.report_comparison import MetricDelta, ReportComparison

from .report_comparability import check_comparable

_METRIC_FIELDS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "environment_actions",
    "repeated_tool_requests",
    "repeated_experiments",
    "hypothesis_revisions",
    "world_model_versions",
    "search_node_expansions",
    "simulations",
    "verification_failures",
    "wall_time_ms",
    "rhae_environment_score",
)
_EXOCORTEX_FIELDS = (
    "reports_generated",
    "reviews_generated",
    "backend_generations",
    "contradictions_raised",
)


def _delta(
    name: str, control: float | int | None, candidate: float | int | None
) -> MetricDelta:
    computed = None if control is None or candidate is None else candidate - control
    return MetricDelta(name=name, control=control, candidate=candidate, delta=computed)


def compare_reports(
    control: BaselineReport, candidate: BaselineReport
) -> ReportComparison:
    deltas = [
        _delta(name, getattr(control.metrics, name), getattr(candidate.metrics, name))
        for name in _METRIC_FIELDS
    ]
    deltas.append(
        _delta(
            "context_ceiling_failures",
            control.context_ceiling_failures,
            candidate.context_ceiling_failures,
        )
    )
    deltas.extend(
        _delta(
            f"exocortex.{name}",
            getattr(control.exocortex, name),
            getattr(candidate.exocortex, name),
        )
        for name in _EXOCORTEX_FIELDS
    )
    return ReportComparison(
        control=control,
        candidate=candidate,
        deltas=tuple(deltas),
        comparability_warnings=check_comparable(control, candidate),
    )
