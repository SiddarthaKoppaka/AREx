"""Exact retrieval preserves ordering and immutable provenance."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.context import EventRetriever
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.retrieval import RetrievalQuery
from arc_agi_3.trace.store import JsonlEventStore


def events(path: Path) -> list[EventEnvelope]:
    store = JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )
    first = store.append(EventType.OBSERVATION, "environment", 0, {"label": "alpha"})
    store.append(
        EventType.ACTION,
        "executor",
        1,
        {"label": "beta"},
        causal_refs=(first.event_id,),
    )
    store.append(EventType.OBSERVATION, "environment", 1, {"label": "gamma"})
    return store.read()


def test_exact_filters_and_provenance_are_deterministic(tmp_path: Path) -> None:
    source = events(tmp_path / "events.jsonl")
    query = RetrievalQuery(event_types=(EventType.OBSERVATION,), order="descending")
    first = EventRetriever(source).retrieve(query)
    second = EventRetriever(source).retrieve(query)
    assert first == second
    assert [item.sequence for item in first.matches] == [2, 0]
    assert first.matches[0].event_hash == source[2].event_hash


def test_literal_and_causal_filters_are_exact(tmp_path: Path) -> None:
    source = events(tmp_path / "events.jsonl")
    literal = EventRetriever(source).retrieve(RetrievalQuery(literal_text="beta"))
    causal = EventRetriever(source).retrieve(
        RetrievalQuery(causal_ref=source[0].event_id)
    )
    assert [item.sequence for item in literal.matches] == [1]
    assert [item.sequence for item in causal.matches] == [1]
