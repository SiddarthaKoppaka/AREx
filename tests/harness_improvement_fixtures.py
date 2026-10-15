"""Shared fixtures for PR-10 harness-improvement tests: three small real
runs (control, a regressing candidate, an improving candidate), reduced
to BaselineReports via PR-00's existing build_report."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.baseline_report import BaselineReport
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EnvironmentState
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.prediction import ExpectedOutcome
from arc_agi_3.research.baseline_report import build_report
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment
from arc_agi_3.testing.epistemic import ReactiveModel


def _correct(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="x",
        intent="x",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        expected_outcome=ExpectedOutcome(state=EnvironmentState.NOT_FINISHED),
    )


def _wrong(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="x",
        intent="x",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        expected_outcome=ExpectedOutcome(state=EnvironmentState.WIN),
    )


def _stop(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(assessment="x", intent="x", mode=DecisionMode.STOP)


def _run(tmp_path: Path, run_id: str, policies: list) -> BaselineReport:  # type: ignore[type-arg]
    config = RunConfig(run_id=run_id, experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config,
        FakeLineEnvironment(target=10),
        ReactiveModel(policies),
        raise_on_failure=True,
    )
    runner.run()
    return build_report(tmp_path / run_id)


def control_report(tmp_path: Path) -> BaselineReport:
    return _run(tmp_path, "control", [_correct, _correct, _stop])


def regressing_candidate_report(tmp_path: Path) -> BaselineReport:
    return _run(tmp_path, "regressing", [_correct, _wrong, _stop])


def improving_candidate_report(tmp_path: Path) -> BaselineReport:
    return _run(tmp_path, "improving", [_correct, _stop])
