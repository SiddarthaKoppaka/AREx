"""A complete run can be re-executed with no live model or environment."""

from pathlib import Path

from arc_agi_3.config import EvaluatorConfig, RunConfig
from arc_agi_3.replay_adapters import RecordedEnvironment, RecordedModel
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def test_recorded_adapters_reproduce_actions_and_metrics(tmp_path: Path) -> None:
    evaluator = EvaluatorConfig(human_action_baselines=(2,))
    source_config = RunConfig(
        run_id="source",
        experiment_id="replay",
        output_dir=tmp_path,
        evaluator=evaluator,
    )
    source = build_runner(
        source_config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    source_result = source.run()
    events = source.events.read()
    replay_config = RunConfig(
        run_id="offline-replay",
        experiment_id="replay",
        output_dir=tmp_path,
        evaluator=evaluator,
    )
    replay = build_runner(
        replay_config, RecordedEnvironment(events), RecordedModel(events)
    )
    replay_result = replay.run()
    assert replay_result.metrics == source_result.metrics
    assert replay_result.stop_reason == source_result.stop_reason
