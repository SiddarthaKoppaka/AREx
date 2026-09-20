"""Exact golden signature plus deterministic metric replay."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.manifest import RunManifest
from arc_agi_3.replay import replay_trace
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def test_context_settings_are_hashed_in_run_provenance(run_config: RunConfig) -> None:
    changed = run_config.model_copy(update={"recent_event_limit": 3})
    assert changed.config_hash != run_config.config_hash
    changed = run_config.model_copy(update={"context_compaction": False})
    assert changed.config_hash != run_config.config_hash


def test_golden_trace_and_replay(
    run_config: RunConfig,
    fixed_clock: Callable[[], datetime],
    event_ids: Callable[[], str],
    fixed_manifest: RunManifest,
) -> None:
    runner = build_runner(
        run_config,
        FakeLineEnvironment(),
        ScriptedModel(successful_script()),
        clock=fixed_clock,
        id_factory=event_ids,
        manifest=fixed_manifest,
    )
    result = runner.run()
    events = runner.events.read()
    signature = {
        "event_hashes": [event.event_hash for event in events],
        "event_types": [event.event_type for event in events],
        "metrics": result.metrics.model_dump(mode="json"),
        "stop_reason": result.stop_reason,
    }
    golden_path = Path(__file__).parent / "golden" / "fake_run_signature.json"
    assert signature == json.loads(golden_path.read_text())
    replay = replay_trace(runner.events.path, run_config.evaluator)
    assert replay.event_count == len(events)
    assert replay.metrics == result.metrics
