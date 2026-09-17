"""Cognitive state flows from LM decision to trace, context, and checkpoint."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import Hypothesis, TaskRecord, ToolRequest
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision, ModelResponse
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment


class CapturingModel:
    def __init__(self, decisions: list[CognitiveDecision]) -> None:
        self.decisions = decisions
        self.contexts: list[AgentContext] = []

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "capturing"}

    def decide(self, context: AgentContext) -> ModelResponse:
        self.contexts.append(context)
        return ModelResponse(decision=self.decisions.pop(0))


def investigate() -> CognitiveDecision:
    hypotheses = (
        Hypothesis(
            hypothesis_id="h1",
            version=1,
            claim="left",
            probability=0.6,
            belief_group="direction",
        ),
        Hypothesis(
            hypothesis_id="h2",
            version=1,
            claim="right",
            probability=0.4,
            belief_group="direction",
        ),
    )
    task = TaskRecord(
        task_id="inspect", version=1, purpose="inspect", success_criteria="evidence"
    )
    tool = ToolRequest(
        request_id="retrieve-1",
        tool_name="retrieve_events",
        arguments={"event_types": ["observation"]},
    )
    return CognitiveDecision(
        assessment="Need exact prior evidence.",
        intent="Record alternatives and retrieve observations.",
        mode=DecisionMode.INVESTIGATE,
        hypothesis_proposals=hypotheses,
        task_updates=(task,),
        tool_requests=(tool,),
    )


def test_workspace_is_visible_and_checkpointed(tmp_path: Path) -> None:
    decisions = [
        investigate(),
        CognitiveDecision(
            assessment="act",
            intent="advance",
            mode=DecisionMode.EXECUTE,
            action=Action(action_id=1),
        ),
        CognitiveDecision(assessment="done", intent="stop", mode=DecisionMode.STOP),
    ]
    model = CapturingModel(decisions)
    config = RunConfig(run_id="cognitive", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(), model)
    result = runner.run()
    events = runner.events.read()
    kinds = {event.event_type for event in events}
    assert result.stop_reason == "agent_stop"
    assert {EventType.HYPOTHESIS, EventType.TASK_UPDATE, EventType.TOOL_RESULT} <= kinds
    assert len(model.contexts[1].hypotheses) == 2
    assert model.contexts[1].tasks[0].task.task_id == "inspect"
    assert EventType.TOOL_RESULT in {
        item.event_type for item in model.contexts[1].recent_events
    }
    checkpoint = runner.io.checkpoints.load("checkpoint-000002")
    hypotheses = checkpoint.cognitive_state["hypotheses"]
    assert isinstance(hypotheses, list) and len(hypotheses) == 2
    assert audit_agency_boundary(events) == ()
