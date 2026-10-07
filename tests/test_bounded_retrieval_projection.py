"""A requested view=full retrieval of a large frame is preserved exactly
in the trace but bounded in automatic context. See the Bounded Retrieval
Projection v1 fix: ADR-0009 follow-up, context/retrieved_projection.py."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from arc_agi_3.context.evidence_retrieval import retrieve_evidence
from arc_agi_3.context.projection import compact_context_event
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.evidence import EvidenceQuery
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.trace.store import JsonlEventStore


def grid_64() -> Observation:
    return Observation.build(
        game_id="grid64",
        frame=[[[(row + col) % 4 for col in range(64)] for row in range(64)]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1],
    )


def store(tmp_path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        tmp_path / "events.jsonl",
        "run",
        "episode",
        "main",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_large_full_view_is_bounded_in_automatic_context_but_exact_in_trace(
    tmp_path: Path,
) -> None:
    events = store(tmp_path)
    observation = grid_64()
    source = events.append(
        EventType.OBSERVATION, "environment", 0, observation.model_dump(mode="json")
    )
    status, output, _ = retrieve_evidence(
        events.read(),
        EvidenceQuery(
            event_ids=(source.event_id,),
            view="full",
            purpose="get the complete grid layout",
            token_budget=65536,
        ),
    )
    assert status == "complete"
    # The tool's own return value is the exact, unbounded frame.
    assert (
        output["items"][0]["content"]["frame"]
        == observation.model_dump(mode="json")["frame"]
    )

    recorded = events.append(
        EventType.TOOL_RESULT,
        "tools",
        0,
        {"result": {"status": status, "output": {"evidence": output}}},
    )
    projected = compact_context_event(recorded)
    visible = cast(Any, projected.payload["result"])
    item = visible["evidence"]["items"][0]
    assert "content" not in item
    assert item["content_projection"]["omitted_from_automatic_context"] is True
    assert item["content_projection"]["size_characters"] > 4096
    assert item["content_projection"]["suggested_views"] == [
        "rle_frame",
        "inspect_frame_region",
    ]
    assert item["event_id"] == source.event_id

    # Nothing is lost from the canonical trace: the exact result is still there.
    stored_result = events.read()[1].payload["result"]["output"]["evidence"]
    assert (
        stored_result["items"][0]["content"]["frame"]
        == observation.model_dump(mode="json")["frame"]
    )
