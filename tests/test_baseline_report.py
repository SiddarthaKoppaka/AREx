"""The baseline report reuses the canonical pipeline with correct missing-vs-zero."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.manifest import RunManifest
from arc_agi_3.research.baseline_report import build_report
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script
from tests.generation_fixtures import ScriptedBackend


def _golden_run_dir(
    run_config: RunConfig,
    fixed_clock: Callable[[], datetime],
    event_ids: Callable[[], str],
    fixed_manifest: RunManifest,
) -> Path:
    build_runner(
        run_config,
        FakeLineEnvironment(),
        ScriptedModel(successful_script()),
        clock=fixed_clock,
        id_factory=event_ids,
        manifest=fixed_manifest,
    ).run()
    return run_config.output_dir / run_config.run_id


def test_same_trace_yields_same_report(
    run_config: RunConfig,
    fixed_clock: Callable[[], datetime],
    event_ids: Callable[[], str],
    fixed_manifest: RunManifest,
) -> None:
    run_dir = _golden_run_dir(run_config, fixed_clock, event_ids, fixed_manifest)
    assert build_report(run_dir) == build_report(run_dir)


def test_baseline_report_matches_golden_run_metrics(
    run_config: RunConfig,
    fixed_clock: Callable[[], datetime],
    event_ids: Callable[[], str],
    fixed_manifest: RunManifest,
) -> None:
    golden_path = Path(__file__).parent / "golden" / "fake_run_signature.json"
    before = golden_path.read_bytes()
    run_dir = _golden_run_dir(run_config, fixed_clock, event_ids, fixed_manifest)
    report = build_report(run_dir)
    golden = json.loads(golden_path.read_text())
    assert report.metrics.model_dump(mode="json") == golden["metrics"]
    assert report.stop_reason == golden["stop_reason"]
    assert golden_path.read_bytes() == before


def test_exocortex_fields_are_missing_when_no_specialist_backend_configured(
    run_config: RunConfig,
) -> None:
    build_runner(
        run_config, FakeLineEnvironment(), ScriptedModel(successful_script())
    ).run()
    report = build_report(run_config.output_dir / run_config.run_id)
    assert report.exocortex.configured is False
    assert report.exocortex.reports_generated is None
    assert report.metrics.specialist_reports_generated == 0


def test_exocortex_fields_are_real_zero_when_specialist_backend_configured_but_unused(
    run_config: RunConfig,
) -> None:
    build_runner(
        run_config,
        FakeLineEnvironment(),
        ScriptedModel(successful_script()),
        specialist_backend=ScriptedBackend([]),
    ).run()
    report = build_report(run_config.output_dir / run_config.run_id)
    assert report.exocortex.configured is True
    assert report.exocortex.reports_generated == 0
