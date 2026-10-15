"""tool_request_evidence surfaces the most recently touched fingerprints,
never silently drops a fresh repeat in favor of a stale first-seen one."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.evidence import tool_fingerprint, tool_request_evidence
from arc_agi_3.trace.store import JsonlEventStore


def store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def request(view: str) -> ToolRequest:
    return ToolRequest(
        request_id=view,
        tool_name="retrieve_evidence",
        arguments={
            "event_ids": ["obs-1"],
            "view": view,
            "purpose": "inspect",
            "token_budget": 1024,
        },
    )


def _append(events: JsonlEventStore, step: int, view: str) -> None:
    events.append(
        EventType.TOOL_REQUEST,
        "tools",
        step,
        {"request": request(view).model_dump(mode="json")},
    )


def test_a_stale_fingerprint_recurring_displaces_the_least_recent_one(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    for step, view in enumerate(f"region-{index}" for index in range(9)):
        _append(events, step, view)
    stale_key = tool_fingerprint(request("region-0"))
    table = tool_request_evidence(events.read(), None)
    assert len(table) == 8
    assert stale_key not in {row.fingerprint for row in table}
    _append(events, 9, "region-0")  # the stale fingerprint recurs
    refreshed = tool_request_evidence(events.read(), None)
    assert len(refreshed) == 8
    assert stale_key in {row.fingerprint for row in refreshed}


def test_an_intervening_unrelated_call_does_not_disturb_attempt_counts(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    _append(events, 0, "a")
    _append(events, 1, "b")
    obs = events.append(EventType.OBSERVATION, "environment", 2, {})
    _append(events, 3, "a")
    table = tool_request_evidence(events.read(), obs.event_id)
    by_key = {row.fingerprint: row for row in table}
    row_a = by_key[tool_fingerprint(request("a"))]
    row_b = by_key[tool_fingerprint(request("b"))]
    assert row_a.attempts == 2
    assert row_a.attempts_from_current_state == 1
    assert row_b.attempts == 1
    assert row_b.attempts_from_current_state == 0
