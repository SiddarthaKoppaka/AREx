"""A fork/recovery restores the full cognitive workspace - scratchpad,
world models, tasks, and hypotheses together - not just environment state.
See PR-02-scoping.md section 6, scenario 5: the existing fork test
(test_recovery.py) only asserts branch/action mechanics, never workspace
content fidelity."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import Hypothesis, TaskRecord
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EnvironmentState, RecoveryOperation
from arc_agi_3.contracts.execution import ExpectedOutcome
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.recovery import RecoveryRequest
from arc_agi_3.contracts.scratchpad import ScratchpadUpdates, VerifiedFact
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment
from arc_agi_3.testing.epistemic import ReactiveModel
from arc_agi_3.world_model.contracts import WorldModel


def author(context: AgentContext) -> CognitiveDecision:
    assert context.current_observation_event_id is not None
    fact = VerifiedFact(
        fact_id="f1",
        version=1,
        fact="learned something real",
        confidence=0.9,
        evidence_refs=(context.current_observation_event_id,),
    )
    return CognitiveDecision(
        assessment="author scratchpad, world model, task, hypothesis",
        intent="advance",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        expected_outcome=ExpectedOutcome(state=EnvironmentState.NOT_FINISHED),
        scratchpad_updates=ScratchpadUpdates(add_verified_fact=(fact,)),
        world_model_updates=(WorldModel(model_id="m1", version=1, rules=()),),
        task_updates=(
            TaskRecord(task_id="t1", version=1, purpose="p", success_criteria="c"),
        ),
        hypothesis_proposals=(
            Hypothesis(hypothesis_id="h1", version=1, claim="x", probability=0.5),
        ),
    )


def execute(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="advance",
        intent="advance",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        expected_outcome=ExpectedOutcome(state=EnvironmentState.NOT_FINISHED),
    )


def fork(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="diverged",
        intent="fork from the authored checkpoint",
        mode=DecisionMode.RECOVER,
        recovery=RecoveryRequest(
            operation=RecoveryOperation.FORK_CHECKPOINT,
            diagnosis="retry from the first checkpoint",
            checkpoint_id="checkpoint-000001",
            new_branch_id="retry",
        ),
    )


def stop(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(assessment="done", intent="stop", mode=DecisionMode.STOP)


def test_fork_restores_scratchpad_world_model_task_and_hypothesis(
    tmp_path: Path,
) -> None:
    model = ReactiveModel([author, execute, fork, stop])
    config = RunConfig(run_id="fidelity", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(target=4), model)
    runner.run()
    restored = model.contexts[-1]
    assert restored.working_scratchpad is not None
    assert any(f.fact_id == "f1" for f in restored.working_scratchpad.verified_facts)
    assert any(m.model_id == "m1" for m in restored.world_models)
    assert any(v.task.task_id == "t1" for v in restored.tasks)
    assert any(h.hypothesis_id == "h1" for h in restored.hypotheses)
