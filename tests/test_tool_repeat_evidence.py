"""Tool-request repeat evidence is mechanical, bounded, and never blocks.
See 'Cognitive Stagnation / Tool Repeat Evidence v1'."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from arc_agi_3.adapters.generation_errors import ToolContractError
from arc_agi_3.adapters.tool_contract_check import check_tool_requests
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.evaluation_tool_repeats import tool_repeat_counts
from arc_agi_3.evidence import tool_request_evidence
from arc_agi_3.trace.store import JsonlEventStore
from tests.epistemic_policies import long_run


def store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def request(request_id: str = "r1") -> ToolRequest:
    return ToolRequest(
        request_id=request_id,
        tool_name="retrieve_evidence",
        arguments={
            "event_ids": ["obs-1"],
            "view": "rle_frame",
            "purpose": "inspect",
            "token_budget": 1024,
        },
    )


def test_duplicate_request_ids_in_one_decision_is_a_contract_violation() -> None:
    decision = CognitiveDecision(
        assessment="x",
        intent="x",
        mode=DecisionMode.INVESTIGATE,
        tool_requests=(request("r1"), request("r1")),
    )
    with pytest.raises(ToolContractError, match="unique request_id"):
        check_tool_requests(decision)


def test_tool_request_evidence_counts_attempts_from_current_state(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    obs1 = events.append(EventType.OBSERVATION, "environment", 0, {})
    events.append(
        EventType.TOOL_REQUEST,
        "tools",
        0,
        {"request": request().model_dump(mode="json")},
    )
    events.append(
        EventType.TOOL_REQUEST,
        "tools",
        1,
        {"request": request("r2").model_dump(mode="json")},
    )
    obs2 = events.append(EventType.OBSERVATION, "environment", 1, {})
    events.append(
        EventType.TOOL_REQUEST,
        "tools",
        2,
        {"request": request("r3").model_dump(mode="json")},
    )
    table = tool_request_evidence(events.read(), obs2.event_id)
    assert len(table) == 1
    row = table[0]
    assert row.attempts == 3
    assert row.attempts_from_current_state == 1
    assert row.tool_name == "retrieve_evidence"
    # Confirm the pre-second-observation attempts are excluded from "current".
    table_before = tool_request_evidence(events.read(), obs1.event_id)
    assert table_before[0].attempts_from_current_state == 2


def test_tool_repeat_counts_flags_only_truly_unjustified_repeats(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    events.append(EventType.OBSERVATION, "environment", 0, {})
    events.append(
        EventType.TOOL_REQUEST,
        "tools",
        0,
        {"request": request("r1").model_dump(mode="json")},
    )
    events.append(
        EventType.TOOL_REQUEST,
        "tools",
        1,
        {"request": request("r2").model_dump(mode="json")},
    )
    events.append(EventType.OBSERVATION, "environment", 1, {})
    events.append(
        EventType.TOOL_REQUEST,
        "tools",
        2,
        {"request": request("r3").model_dump(mode="json")},
    )
    counts = tool_repeat_counts(events.read())
    assert counts["tool_requests"] == 3
    assert counts["unique_tool_requests"] == 1
    assert counts["repeated_tool_requests"] == 2
    # Only the first repeat (step 1, same observation) is unjustified; the
    # second repeat (step 2) followed a genuine observation change.
    assert counts["unjustified_tool_repeats"] == 1


def test_core_agent_sees_repeats_from_an_unchanged_observation(
    tmp_path: Path,
) -> None:
    investigate = CognitiveDecision(
        assessment="check",
        intent="check again",
        mode=DecisionMode.INVESTIGATE,
        tool_requests=(request("probe"),),
    )
    _, model = long_run(tmp_path, [lambda _: investigate, lambda _: investigate])
    late = model.contexts[-1]
    assert late.tool_evidence[0].attempts == 2
    assert late.tool_evidence[0].attempts_from_current_state == 2
