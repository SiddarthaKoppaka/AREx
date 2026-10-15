"""Documents a confirmed, narrow limitation (PR-02-scoping.md section 2 /
section 4-E, left as an open question, not fixed here): a fork/recovery
only restores the *current* version of each world model, so an older,
non-current version becomes unreachable afterward even though its content
is still in the immutable trace."""

from pathlib import Path

import pytest

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EnvironmentState, RecoveryOperation
from arc_agi_3.contracts.execution import ExpectedOutcome
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.recovery import RecoveryRequest
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment
from arc_agi_3.testing.epistemic import Policy, ReactiveModel
from arc_agi_3.world_model.contracts import WorldModel


def author_version(version: int) -> Policy:
    def decide(context: AgentContext) -> CognitiveDecision:
        return CognitiveDecision(
            assessment=f"author v{version}",
            intent="advance",
            mode=DecisionMode.EXECUTE,
            action=Action(action_id=1),
            expected_outcome=ExpectedOutcome(state=EnvironmentState.NOT_FINISHED),
            world_model_updates=(WorldModel(model_id="m1", version=version, rules=()),),
        )

    return decide


def fork(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="diverged",
        intent="fork from after v2 was authored",
        mode=DecisionMode.RECOVER,
        recovery=RecoveryRequest(
            operation=RecoveryOperation.FORK_CHECKPOINT,
            diagnosis="retry from the latest checkpoint",
            checkpoint_id="checkpoint-000002",
            new_branch_id="retry",
        ),
    )


def stop(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(assessment="done", intent="stop", mode=DecisionMode.STOP)


def test_an_older_world_model_version_is_unreachable_after_a_fork(
    tmp_path: Path,
) -> None:
    model = ReactiveModel([author_version(1), author_version(2), fork, stop])
    config = RunConfig(run_id="versions", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(target=4), model)
    runner.run()
    # Confirmed limitation: only the current (v2) version survived the fork.
    history = runner.io.workspace.world_models.history("m1")
    assert [item.version for item in history] == [2]
    assert runner.io.workspace.world_models.get("m1", 2).version == 2
    with pytest.raises(KeyError):
        runner.io.workspace.world_models.get("m1", 1)
