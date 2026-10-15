"""Comparison deltas and Markdown rendering preserve missing-vs-zero."""

from pathlib import Path

from arc_agi_3.adapters.inference import InferenceBackend
from arc_agi_3.config import RunConfig
from arc_agi_3.research.baseline_report import build_report
from arc_agi_3.research.report_comparison import compare_reports
from arc_agi_3.research.report_markdown import render_markdown
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script
from tests.generation_fixtures import ScriptedBackend


def _run(
    run_config: RunConfig, *, specialist_backend: InferenceBackend | None = None
) -> Path:
    build_runner(
        run_config,
        FakeLineEnvironment(),
        ScriptedModel(successful_script()),
        specialist_backend=specialist_backend,
    ).run()
    return run_config.output_dir / run_config.run_id


def test_compare_reports_computes_deltas_for_two_differing_runs(
    run_config: RunConfig, tmp_path: Path
) -> None:
    control = build_report(_run(run_config))
    candidate_config = run_config.model_copy(
        update={"run_id": "candidate-run", "output_dir": tmp_path}
    )
    candidate = build_report(_run(candidate_config))
    comparison = compare_reports(control, candidate)
    by_name = {delta.name: delta for delta in comparison.deltas}
    assert by_name["model_calls"].delta == (
        candidate.metrics.model_calls - control.metrics.model_calls
    )
    assert by_name["environment_actions"].delta == (
        candidate.metrics.environment_actions - control.metrics.environment_actions
    )


def test_compare_reports_preserves_missing_not_zero_when_sides_disagree_on_exocortex(
    run_config: RunConfig, tmp_path: Path
) -> None:
    control = build_report(_run(run_config))
    candidate_config = run_config.model_copy(
        update={"run_id": "candidate-run", "output_dir": tmp_path}
    )
    candidate = build_report(
        _run(candidate_config, specialist_backend=ScriptedBackend([]))
    )
    comparison = compare_reports(control, candidate)
    by_name = {delta.name: delta for delta in comparison.deltas}
    assert by_name["exocortex.reports_generated"].delta is None


def test_render_markdown_shows_not_observed_for_missing_values(
    run_config: RunConfig,
) -> None:
    control = build_report(_run(run_config))
    comparison = compare_reports(control, control)
    rendered = render_markdown(comparison)
    assert "not observed" in rendered
    assert "None" not in rendered
    assert "null" not in rendered


def test_render_markdown_is_deterministic(run_config: RunConfig) -> None:
    control = build_report(_run(run_config))
    comparison = compare_reports(control, control)
    assert render_markdown(comparison) == render_markdown(comparison)
