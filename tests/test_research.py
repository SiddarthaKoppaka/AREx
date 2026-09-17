"""Research index, export, and aggregate regression tests."""

import json
from pathlib import Path

from arc_agi_3.research import TraceIndex, aggregate_rows, export_rows, run_row
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def test_index_is_rebuilt_from_verified_raw_trace(run_config) -> None:  # type: ignore[no-untyped-def]
    runner = build_runner(
        run_config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    runner.run()
    index = TraceIndex(run_config.output_dir / "derived.sqlite3")
    assert index.rebuild(runner.events.path) == len(runner.events.read())
    assert index.search("executor")
    index.path.unlink()
    assert index.rebuild(runner.events.path) == len(runner.events.read())


def test_export_has_trace_and_config_provenance(run_config) -> None:  # type: ignore[no-untyped-def]
    runner = build_runner(
        run_config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    runner.run()
    row = run_row(run_config.output_dir / run_config.run_id)
    assert row["config_hash"] == run_config.config_hash
    assert row["last_event_hash"] == runner.events.last_hash
    output = run_config.output_dir / "runs.jsonl"
    export_rows([row], output)
    assert json.loads(output.read_text())["run_id"] == run_config.run_id


def test_seeded_aggregate_is_repeatable() -> None:
    rows: list[dict[str, object]] = [
        {"model": "m", "score": 0.2},
        {"model": "m", "score": 0.8},
    ]
    first = aggregate_rows(rows, ("model",), "score", seed=7, bootstrap_samples=100)
    second = aggregate_rows(rows, ("model",), "score", seed=7, bootstrap_samples=100)
    assert first == second
    assert first[0].mean == 0.5


def test_csv_export_serializes_nested_values(tmp_path: Path) -> None:
    output = tmp_path / "rows.csv"
    export_rows([{"run": "a", "nested": {"enabled": True}}], output)
    assert '""enabled"": true' in output.read_text()
