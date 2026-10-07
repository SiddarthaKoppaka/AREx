"""A structured-output failure is a first-class, diagnosable recoverable event."""

import json
from pathlib import Path

from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, successful_script
from tests.generation_fixtures import ScriptedBackend

VALID = json.dumps(successful_script()[0].model_dump(mode="json"))


def test_failure_event_reports_turn_pending_action_and_categories(
    tmp_path: Path,
) -> None:
    backend = ScriptedBackend(["x", "y", "z"])
    config = RunConfig(run_id="fail-event", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), StructuredModelAdapter(backend, max_repairs=2)
    )
    result = runner.run()
    assert result.stop_reason == "failure"
    failure = next(e for e in runner.events.read() if e.event_type is EventType.FAILURE)
    payload = failure.payload
    # Turn 1 was genuinely in progress (its MODEL_CALLS spend was already
    # recorded) when the exhausted repair loop crashed, not turn 0.
    assert payload["turn"] == 1
    assert payload["environment_action_pending"] is False
    assert payload["checkpoint_available"] is False
    assert payload["last_decision_event_id"] is None
    assert payload["error_categories"] == ["no_json_found"]


def test_failure_event_points_at_the_last_good_decision_and_checkpoint(
    tmp_path: Path,
) -> None:
    backend = ScriptedBackend([VALID, "bad", "bad", "bad"])
    config = RunConfig(
        run_id="fail-after-success",
        experiment_id="test",
        output_dir=tmp_path,
        checkpoint_every=1,
    )
    runner = build_runner(
        config, FakeLineEnvironment(), StructuredModelAdapter(backend, max_repairs=2)
    )
    result = runner.run()
    assert result.stop_reason == "failure"
    events = runner.events.read()
    decision = next(e for e in events if e.event_type is EventType.MODEL_DECISION)
    failure = next(e for e in events if e.event_type is EventType.FAILURE)
    assert failure.payload["last_decision_event_id"] == decision.event_id
    assert failure.payload["checkpoint_available"] is True
    # Turn 1 completed successfully (self.turn advanced to 1); the crash
    # happened during turn 2, which must not be misattributed to turn 1.
    assert failure.payload["turn"] == 2
