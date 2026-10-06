"""Shared fixtures for consult_specialist/compare_specialist_reports tests."""

import json

from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode

HYPOTHESIS_REPORT = json.dumps(
    {
        "specialist_id": "S-hypothesis",
        "faculty": "hypothesis",
        "turn": 1,
        "assessment": "ok",
    }
)


def investigate(tool_name: str, arguments: dict[str, object]) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="x",
        intent="x",
        mode=DecisionMode.INVESTIGATE,
        tool_requests=(
            ToolRequest(request_id="r1", tool_name=tool_name, arguments=arguments),
        ),
    )


def stop() -> CognitiveDecision:
    return CognitiveDecision(assessment="x", intent="x", mode=DecisionMode.STOP)
