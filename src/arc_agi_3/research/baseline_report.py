"""Build a `BaselineReport` by reusing the canonical trace/evaluator pipeline."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.contracts.baseline_report import BaselineReport, ExocortexCoverage
from arc_agi_3.contracts.evaluation import EvaluationMetrics
from arc_agi_3.evaluation_report_fields import context_ceiling_failures, stop_reason

from .export import load_run


def _exocortex_coverage(
    model: dict[str, JsonValue], metrics: EvaluationMetrics
) -> ExocortexCoverage:
    if "exocortex" not in model:
        return ExocortexCoverage(configured=False)
    return ExocortexCoverage(
        configured=True,
        reports_generated=metrics.specialist_reports_generated,
        reviews_generated=metrics.specialist_reviews_generated,
        backend_generations=metrics.exocortex_backend_generations,
        contradictions_raised=metrics.flagged_contradictions_raised,
    )


def build_report(run_dir: Path) -> BaselineReport:
    manifest, events, metrics = load_run(run_dir)
    seed = manifest.config.get("seed", 0)
    ablations = manifest.config.get("ablations", {})
    return BaselineReport(
        run_id=manifest.run_id,
        experiment_id=manifest.experiment_id,
        game_id=str(manifest.config.get("game_id", "")),
        seed=seed if isinstance(seed, int) else 0,
        config_hash=manifest.config_hash,
        last_event_hash=events[-1].event_hash if events else "",
        event_count=len(events),
        model=manifest.model,
        environment=manifest.environment,
        ablations=ablations if isinstance(ablations, dict) else {},
        stop_reason=stop_reason(events),
        context_ceiling_failures=context_ceiling_failures(events),
        exocortex=_exocortex_coverage(manifest.model, metrics),
        metrics=metrics,
    )
