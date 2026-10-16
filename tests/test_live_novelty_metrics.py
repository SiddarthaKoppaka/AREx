"""PR-12 (idea B) + this morning's no-novelty streak: both are purely
mechanical, trace-derived counters read live into `ContextStats` - the
model sees them, nothing acts on them. Neither had a direct unit test
before this file."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.evaluation_tool_repeats import (
    current_no_novelty_streak,
    tool_requests_since_last_action,
)
from arc_agi_3.trace.store import JsonlEventStore


def store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def _request(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "tool_name": "retrieve_evidence",
        "arguments": {
            "event_ids": ["obs-1"],
            "view": "rle_frame",
            "purpose": "inspect",
            "token_budget": 1024,
        },
    }


def test_tool_requests_since_last_action_is_zero_at_episode_start(
    tmp_path: Path,
) -> None:
    assert tool_requests_since_last_action([]) == 0


def test_tool_requests_since_last_action_counts_up_then_resets_on_action(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    events.append(EventType.OBSERVATION, "environment", 0, {})
    events.append(EventType.TOOL_REQUEST, "tools", 0, {"request": _request("r1")})
    events.append(EventType.TOOL_REQUEST, "tools", 1, {"request": _request("r2")})
    assert tool_requests_since_last_action(events.read()) == 2

    events.append(
        EventType.ACTION,
        "executor",
        2,
        {"action": {"action_id": 1}, "before_hash": "x"},
    )
    assert tool_requests_since_last_action(events.read()) == 0

    events.append(EventType.TOOL_REQUEST, "tools", 3, {"request": _request("r3")})
    assert tool_requests_since_last_action(events.read()) == 1


def test_current_no_novelty_streak_is_zero_at_episode_start(tmp_path: Path) -> None:
    assert current_no_novelty_streak([]) == 0


def test_current_no_novelty_streak_counts_consecutive_repeat_only_turns(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    events.append(EventType.OBSERVATION, "environment", 0, {})
    events.append(EventType.TOOL_REQUEST, "tools", 0, {"request": _request("r1")})
    assert current_no_novelty_streak(events.read()) == 0

    events.append(EventType.TOOL_REQUEST, "tools", 1, {"request": _request("r2")})
    events.append(EventType.TOOL_REQUEST, "tools", 2, {"request": _request("r3")})
    assert current_no_novelty_streak(events.read()) == 2

    events.append(EventType.OBSERVATION, "environment", 3, {})
    events.append(EventType.TOOL_REQUEST, "tools", 3, {"request": _request("r4")})
    assert current_no_novelty_streak(events.read()) == 0
