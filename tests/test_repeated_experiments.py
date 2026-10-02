"""Equivalent experiments are flagged as cues, never blocked."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.prediction import ExpectedOutcome, Experiment
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.state import RunResult
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, ReactiveModel
from tests.epistemic_policies import stop


def act(action_id: int, why: str | None = None, **claims: object) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="Probe the action.",
        intent="Collect transition evidence.",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=action_id),
        experiment=Experiment(
            experiment_id="probe", question="What changes?", repeat_justification=why
        ),
        expected_outcome=ExpectedOutcome.model_validate(claims),
    )


Run = tuple[RunResult, list[dict[str, JsonValue]], list[AgentContext]]


def run(tmp_path: Path, decisions: list[CognitiveDecision]) -> Run:
    policies = [lambda _, d=d: d for d in decisions]
    model = ReactiveModel([*policies, stop])
    config = RunConfig(run_id="repeat", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    result = runner.run()
    experiments = [
        e.payload for e in runner.events.read() if e.event_type is EventType.EXPERIMENT
    ]
    return result, experiments, model.contexts


def test_repeat_is_flagged_and_justified_repeat_is_allowed(tmp_path: Path) -> None:
    decisions = [
        act(2, no_op=True),
        act(2, no_op=True),
        act(2, "Checking for a delayed effect after two steps.", no_op=True),
    ]
    result, experiments, contexts = run(tmp_path, decisions)
    assert result.metrics.environment_actions == 3
    assert [e["unjustified_repeat"] for e in experiments] == [False, True, False]
    prior = experiments[1]["repeat"]["equivalent_prior"]
    assert [item["match"] for item in prior] == ["same_state_same_action"]
    assert (
        prior[0]["changed_cells"] == 0 and prior[0]["verification_status"] == "matched"
    )
    metrics = result.metrics
    assert (metrics.repeated_experiments, metrics.unjustified_repeats) == (2, 1)
    cue = next(a for a in contexts[1].action_evidence if a.action.action_id == 2)
    assert (cue.attempts, cue.attempts_from_current_state) == (1, 1)
    assert cue.recent_outcomes[0].changed_cells == 0


def test_same_prediction_from_a_new_state_is_flagged(tmp_path: Path) -> None:
    claims = {"min_changed_cells": 1}
    _, experiments, contexts = run(tmp_path, [act(1, **claims), act(1, **claims)])
    prior = experiments[1]["repeat"]["equivalent_prior"]
    assert [item["match"] for item in prior] == ["same_action_same_prediction"]
    cue = next(a for a in contexts[1].action_evidence if a.action.action_id == 1)
    assert cue.attempts_from_current_state == 0
    assert "value 3 at [2, 1] shifted [0, 1]" in cue.recent_outcomes[0].translations
