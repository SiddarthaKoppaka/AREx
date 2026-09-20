"""Expected budget stops and failed-generation accounting."""

from pathlib import Path

from arc_agi_3.config import BudgetConfig, RunConfig
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType, Resource
from arc_agi_3.contracts.execution import ActionChunk, ChunkStep
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.runner import EpisodeRunner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def config(tmp_path: Path, run_id: str, **limits: int) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment_id="budgets",
        output_dir=tmp_path,
        budget=BudgetConfig(
            limits={Resource(key): value for key, value in limits.items()}
        ),
    )


def action() -> CognitiveDecision:
    return CognitiveDecision(
        assessment="advance",
        intent="advance",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
    )


def assert_clean_stop(runner: EpisodeRunner, reason: str) -> None:
    result = runner.run()
    kinds = [event.event_type for event in runner.events.read()]
    assert result.stop_reason == reason
    assert EventType.RUN_FINISHED in kinds
    assert EventType.FAILURE not in kinds


def test_model_call_exhaustion_is_clean(tmp_path: Path) -> None:
    runner = build_runner(
        config(tmp_path, "call-stop", model_calls=1, actions=4),
        FakeLineEnvironment(),
        ScriptedModel([action()]),
    )
    assert_clean_stop(runner, "model_call_budget_exhausted")
    assert runner.session.result.metrics.model_calls == 1
    assert runner.session.result.metrics.environment_actions == 1


def test_direct_action_exhaustion_is_clean(tmp_path: Path) -> None:
    runner = build_runner(
        config(tmp_path, "action-stop", model_calls=4, actions=1),
        FakeLineEnvironment(),
        ScriptedModel([action()]),
    )
    assert_clean_stop(runner, "action_budget_exhausted")


def test_chunk_action_exhaustion_is_clean(tmp_path: Path) -> None:
    chunk = ActionChunk(
        chunk_id="bounded",
        steps=(
            ChunkStep(action=Action(action_id=1)),
            ChunkStep(action=Action(action_id=1)),
        ),
    )
    choice = CognitiveDecision(
        assessment="advance",
        intent="advance",
        mode=DecisionMode.EXECUTE,
        action_chunk=chunk,
    )
    runner = build_runner(
        config(tmp_path, "chunk-stop", model_calls=4, actions=1),
        FakeLineEnvironment(),
        ScriptedModel([choice]),
    )
    assert_clean_stop(runner, "action_budget_exhausted")
    assert runner.session.result.metrics.environment_actions == 1
