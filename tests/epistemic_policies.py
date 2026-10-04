"""Shared LM test-double policies; they read only generic context fields."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import Hypothesis
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.prediction import (
    ExpectedOutcome,
    Experiment,
    TranslationExpectation,
)
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.runner import EpisodeRunner
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, Policy, ReactiveModel


def probe(hypothesis: str, value: int) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="Test whether a component of this value shifts.",
        intent="Run a discriminating experiment.",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        experiment=Experiment(
            experiment_id=f"e-{hypothesis}",
            question=f"Does value {value} shift?",
            hypothesis_ids=(hypothesis,),
        ),
        expected_outcome=ExpectedOutcome(
            prediction_id=f"p-{hypothesis}",
            hypothesis_ids=(hypothesis,),
            translations=(TranslationExpectation(value=value, d_row=0, d_col=1),),
        ),
    )


def hypothesize(context: AgentContext) -> CognitiveDecision:
    claim = Hypothesis(
        hypothesis_id="h1", version=1, claim="Value 1 is controllable.", probability=0.7
    )
    return probe("h1", 1).model_copy(update={"hypothesis_proposals": (claim,)})


def stop(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(assessment="done", intent="stop", mode=DecisionMode.STOP)


TURNS = 14


def step(index: int) -> Policy:
    def decide(context: AgentContext) -> CognitiveDecision:
        proposals = (
            Hypothesis(hypothesis_id="h1", version=1, claim="c", probability=0.5),
        )
        return CognitiveDecision(
            assessment="Probe.",
            intent="Gather evidence.",
            mode=DecisionMode.EXECUTE,
            action=Action(action_id=1 + index % 2),
            expected_outcome=ExpectedOutcome(
                hypothesis_ids=("h1",), max_changed_cells=0
            ),
            hypothesis_proposals=proposals if index == 0 else (),
        )

    return decide


def long_run(tmp_path: Path, tail: list[Policy]) -> tuple[EpisodeRunner, ReactiveModel]:
    model = ReactiveModel([*(step(i) for i in range(TURNS)), *tail, stop])
    config = RunConfig(
        run_id="long", experiment_id="test", output_dir=tmp_path, max_turns=40
    )
    config.budget.limits.update({"actions": 40, "model_calls": 40})
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    runner.run()
    return runner, model
