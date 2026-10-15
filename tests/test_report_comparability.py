"""PR-11 (cross-cutting evaluation protocol): never silently compare two
runs that aren't actually matched (game_id/seed/model/environment/
ablations), and always state the sample size/uncertainty honestly."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.research.baseline_report import build_report
from arc_agi_3.research.report_comparison import compare_reports
from arc_agi_3.research.report_markdown import render_markdown
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script
from tests.harness_improvement_fixtures import (
    control_report,
    improving_candidate_report,
)


def _report_with_game_id(tmp_path: Path, run_id: str, game_id: str):  # type: ignore[no-untyped-def]
    config = RunConfig(
        run_id=run_id, experiment_id="test", output_dir=tmp_path, game_id=game_id
    )
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    runner.run()
    return build_report(tmp_path / run_id)


def test_matched_runs_produce_no_comparability_warnings(tmp_path: Path) -> None:
    comparison = compare_reports(
        control_report(tmp_path), improving_candidate_report(tmp_path)
    )
    assert comparison.comparability_warnings == ()


def test_a_mismatched_game_id_is_flagged_not_silently_compared(tmp_path: Path) -> None:
    a = _report_with_game_id(tmp_path / "a", "r", "game-a")
    b = _report_with_game_id(tmp_path / "b", "r", "game-b")
    comparison = compare_reports(a, b)
    assert any("game_id" in w for w in comparison.comparability_warnings)


def test_markdown_surfaces_warnings_and_the_sample_size_note(tmp_path: Path) -> None:
    a = _report_with_game_id(tmp_path / "a", "r", "game-a")
    b = _report_with_game_id(tmp_path / "b", "r", "game-b")
    rendered = render_markdown(compare_reports(a, b))
    assert "Comparability warnings" in rendered
    assert "n=1 run per side" in rendered
