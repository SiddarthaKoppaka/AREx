"""Append-only hash-chain integrity."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.trace.store import JsonlEventStore, TraceIntegrityError


def store(path: Path) -> JsonlEventStore:
    ids = iter(("one", "two"))
    return JsonlEventStore(
        path,
        "run",
        "episode",
        "main",
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        id_factory=lambda: next(ids),
    )


def test_trace_round_trip(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    first = events.append(EventType.RUN_STARTED, "test", 0, {"value": 1})
    second = events.append(
        EventType.RUN_FINISHED,
        "test",
        1,
        {"value": 2},
        causal_refs=(first.event_id,),
    )
    assert events.read() == [first, second]


def test_tampering_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    events = store(path)
    events.append(EventType.RUN_STARTED, "test", 0, {"value": 1})
    path.write_text(path.read_text().replace('"value":1', '"value":9'))
    with pytest.raises(TraceIntegrityError):
        events.read()
