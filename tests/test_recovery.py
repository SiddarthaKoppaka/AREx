"""LM-selected rollback forks history without refunding spent resources."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import (
    AgentContext,
    CognitiveDecision,
    ExpectedOutcome,
    ModelResponse,
)
from arc_agi_3.contracts.enums import (
    DecisionMode,
    EnvironmentState,
    EventType,
    RecoveryOperation,
    Resource,
)
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.recovery import RecoveryRequest
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment


class CapturingModel:
    def __init__(self, decisions: list[CognitiveDecision]) -> None:
        self.decisions = decisions
        self.contexts: list[AgentContext] = []

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "recovery-script"}

    def decide(self, context: AgentContext) -> ModelResponse:
        self.contexts.append(context)
        return ModelResponse(decision=self.decisions.pop(0))


def execute(expected: EnvironmentState) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="authorized fixture step",
        intent="advance",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        expected_outcome=ExpectedOutcome(state=expected),
    )


def test_checkpoint_fork_restores_state_and_preserves_failed_branch(
    tmp_path: Path,
) -> None:
    recover = CognitiveDecision(
        assessment="The previous prediction diverged.",
        intent="Fork from the LM-selected earlier checkpoint.",
        mode=DecisionMode.RECOVER,
        recovery=RecoveryRequest(
            operation=RecoveryOperation.FORK_CHECKPOINT,
            diagnosis="LM diagnosis: retry from before the mismatch.",
            checkpoint_id="checkpoint-000001",
            new_branch_id="retry",
        ),
    )
    stop = CognitiveDecision(
        assessment="restored", intent="stop", mode=DecisionMode.STOP
    )
    model = CapturingModel(
        [
            execute(EnvironmentState.NOT_FINISHED),
            execute(EnvironmentState.WIN),
            recover,
            stop,
        ]
    )
    config = RunConfig(run_id="recovery", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(target=4), model)
    result = runner.run()
    events = runner.events.read()
    assert result.stop_reason == "agent_stop"
    assert model.contexts[2].recovery_evidence is not None
    assert model.contexts[3].observation.frame[0][0][1] == 2
    assert result.metrics.environment_actions == 2
    assert runner.io.ledger.snapshot().consumed[Resource.MODEL_CALLS] == 4
    assert runner.io.branches.active_branch_id == "retry"
    assert {event.branch_id for event in events} == {"main", "retry"}
    assert EventType.BRANCH_FROZEN in {event.event_type for event in events}
    assert audit_agency_boundary(events) == ()
