"""Model-authored task completion persists a compact evidence handoff."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import TaskHandoff, TaskRecord
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision, ModelResponse
from arc_agi_3.contracts.enums import DecisionMode, TaskStatus
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment


class HandoffModel:
    calls = 0

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "handoff-test"}

    def decide(self, context: AgentContext) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                decision=CognitiveDecision(
                    assessment="A movement test is needed.",
                    intent="Test one action.",
                    mode=DecisionMode.EXECUTE,
                    action=Action(action_id=1),
                    task_updates=(
                        TaskRecord(
                            task_id="action-test",
                            version=1,
                            purpose="Discover action 1 effect",
                            success_criteria="Observe one resulting transition",
                        ),
                    ),
                )
            )
        reference = context.recent_event_refs[-1]
        return ModelResponse(
            decision=CognitiveDecision(
                assessment="The transition is recorded.",
                intent="Complete this subtask with evidence.",
                mode=DecisionMode.STOP,
                task_updates=(
                    TaskRecord(
                        task_id="action-test",
                        version=2,
                        purpose="Discover action 1 effect",
                        success_criteria="Observe one resulting transition",
                        status=TaskStatus.COMPLETE,
                        handoff=TaskHandoff(
                            summary="Action 1 was tested once.",
                            evidence_refs=(reference,),
                        ),
                    ),
                ),
            )
        )


def test_handoff_is_model_authored_and_visible_in_workspace(tmp_path: Path) -> None:
    model = HandoffModel()
    config = RunConfig(run_id="handoff", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(), model)
    result = runner.run()
    assert result.stop_reason == "agent_stop"
    assert model.calls == 2
    task = runner.io.workspace.tasks.current[0]
    assert task.status is TaskStatus.COMPLETE
    assert task.handoff is not None
    assert task.handoff.evidence_refs[0] in {
        event.event_id for event in runner.events.read()
    }
