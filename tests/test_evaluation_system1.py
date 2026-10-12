"""system1_counts is purely mechanical: direct reads of recorded fields."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.evaluation_system1 import system1_counts
from arc_agi_3.trace.store import JsonlEventStore
from tests.system1_fixtures import result_selecting, two_candidate_request


def store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_counts_are_zero_with_no_decisions(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    counts = system1_counts(events.read())
    assert counts["system1_requests"] == 0
    assert counts["system1_mean_confidence"] == 0.0


def test_counts_reflect_recorded_decisions(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    result = result_selecting("move-down")
    events.append(
        EventType.SYSTEM1_DECISION,
        "exocortex",
        1,
        {
            "request": two_candidate_request(),
            "result": result.model_dump(mode="json"),
        },
    )
    counts = system1_counts(events.read())
    assert counts["system1_requests"] == 1
    assert counts["system1_suggestions"] == 1
    assert counts["system1_delegated_executions"] == 0
    assert counts["system1_mean_candidate_count"] == 2.0
    assert counts["system1_mean_confidence"] == 0.8
    assert counts["system1_stall_detections"] == 1
    assert counts["system1_mean_entropy"] > 0.0
