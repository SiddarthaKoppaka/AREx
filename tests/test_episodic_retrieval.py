"""Episodic retrieval deterministically favors objective-relevant summaries."""

from datetime import UTC, datetime

from arc_agi_3.context import summarize_episode
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.trace.store import JsonlEventStore


def test_objective_terms_select_relevant_older_episode(tmp_path) -> None:
    events = JsonlEventStore(
        tmp_path / "events.jsonl",
        "run",
        "episode",
        "main",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    for step, summary in enumerate(("inspect target door", "move", "wait")):
        events.append(
            EventType.MODEL_DECISION,
            "model",
            step,
            {"decision": {"decision_summary": summary}},
        )
    first = summarize_episode(events.read(), item_limit=1, query="target")
    second = summarize_episode(events.read(), item_limit=1, query="target")
    assert first == second
    assert first is not None
    assert first.items[0].step_id == 0
    assert first.items[0].summary == "inspect target door"


def test_zero_limit_disables_episodic_retrieval(tmp_path) -> None:
    events = JsonlEventStore(tmp_path / "events.jsonl", "run", "episode", "main")
    assert summarize_episode(events.read(), item_limit=0) is None
