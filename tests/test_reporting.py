"""Live reporting is observational and preserves the canonical trace."""

from pathlib import Path

import pytest

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.reporting import LiveReporter
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def test_reporter_observes_events_without_changing_trace(tmp_path: Path) -> None:
    reporter = LiveReporter("readable", tmp_path / "console.log")
    config = RunConfig(run_id="report", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config,
        FakeLineEnvironment(),
        ScriptedModel(successful_script()),
        reporter=reporter,
    )
    result = runner.run()
    assert result.stop_reason == "terminal"
    assert runner.events.read()[-1].event_type is EventType.EVALUATION
    log = (tmp_path / "console.log").read_text()
    assert "assessment" in log and "intent" in log
    assert "frame_dimensions" in log and "evaluation" in log


def test_raise_on_failure_preserves_trace(tmp_path: Path) -> None:
    config = RunConfig(run_id="raise", experiment_id="test", output_dir=tmp_path)
    invalid = CognitiveDecision(
        assessment="bad",
        intent="act",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=7),
    )
    runner = build_runner(
        config,
        FakeLineEnvironment(),
        ScriptedModel([invalid]),
        raise_on_failure=True,
    )
    with pytest.raises(ValueError, match="not currently authorized"):
        runner.run()
    events = runner.events.read()
    assert EventType.FAILURE in [event.event_type for event in events]
    assert events[-1].event_type is EventType.EVALUATION


def test_silent_and_json_reporting_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    silent_path = tmp_path / "silent.log"
    LiveReporter("silent", silent_path).stage("generation_start")
    assert capsys.readouterr().out == ""
    assert not silent_path.exists()
    config = RunConfig(run_id="json", experiment_id="test", output_dir=tmp_path)
    reporter = LiveReporter("json", tmp_path / "json.log")
    runner = build_runner(
        config,
        FakeLineEnvironment(),
        ScriptedModel(successful_script()),
        reporter=reporter,
    )
    runner.run()
    assert '"event_type":"evaluation"' in (tmp_path / "json.log").read_text()
