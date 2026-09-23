"""Missing artifact references remain bounded tool failures."""

from pathlib import Path
from typing import Any, cast

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def test_missing_artifact_records_failure_and_runtime_continues(
    tmp_path: Path,
) -> None:
    missing = f"artifacts/{'0' * 64}.json"
    decisions = [
        CognitiveDecision(
            assessment="A prior artifact may contain useful evidence.",
            intent="Try to reopen it.",
            mode=DecisionMode.INVESTIGATE,
            tool_requests=(
                ToolRequest(
                    request_id="missing-artifact",
                    tool_name="retrieve_artifact",
                    arguments={
                        "artifact_ref": missing,
                        "purpose": "Resume the prior phase",
                        "token_budget": 512,
                    },
                ),
            ),
        ),
        CognitiveDecision(
            assessment="The artifact is unavailable.",
            intent="Stop cleanly.",
            mode=DecisionMode.STOP,
        ),
    ]
    runner = build_runner(
        RunConfig(run_id="missing-artifact", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(),
        ScriptedModel(decisions),
    )
    result = runner.run()
    tool_results = [
        event
        for event in runner.events.read()
        if event.event_type is EventType.TOOL_RESULT
    ]
    assert result.stop_reason == "agent_stop"
    assert len(tool_results) == 1
    payload = cast(dict[str, Any], tool_results[0].payload["result"])
    assert payload["status"] == "failed"
    assert payload["output"]["error_type"] == "FileNotFoundError"
