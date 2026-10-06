"""The core agent's scripted decision sequence for the ExoCortex e2e test."""

from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision, ModelResponse
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.observation import Action


class RecordingCoreAgent:
    metadata = {"provider": "test"}

    def __init__(self, decisions: list[CognitiveDecision]) -> None:
        self.decisions = list(decisions)
        self.contexts: list[AgentContext] = []

    def decide(self, context: AgentContext) -> ModelResponse:
        self.contexts.append(context)
        return ModelResponse(decision=self.decisions.pop(0))


def three_turn_core_agent() -> RecordingCoreAgent:
    return RecordingCoreAgent(
        [
            CognitiveDecision(
                assessment="need another view on value 11",
                intent="consult hypothesis faculty",
                mode=DecisionMode.INVESTIGATE,
                tool_requests=(
                    ToolRequest(
                        request_id="r1",
                        tool_name="consult_specialist",
                        arguments={"faculty": "hypothesis"},
                    ),
                ),
            ),
            CognitiveDecision(
                assessment="check that reading",
                intent="consult critic faculty",
                mode=DecisionMode.INVESTIGATE,
                tool_requests=(
                    ToolRequest(
                        request_id="r2",
                        tool_name="consult_specialist",
                        arguments={"faculty": "critic"},
                    ),
                ),
            ),
            CognitiveDecision(
                assessment="compare the two faculties",
                intent="resolve disagreement",
                mode=DecisionMode.INVESTIGATE,
                tool_requests=(
                    ToolRequest(
                        request_id="r3",
                        tool_name="compare_specialist_reports",
                        arguments={
                            "primary_faculty": "critic",
                            "other_faculties": ["hypothesis"],
                        },
                    ),
                ),
            ),
            CognitiveDecision(
                assessment="advance",
                intent="finish",
                mode=DecisionMode.EXECUTE,
                action=Action(action_id=1),
            ),
        ]
    )
