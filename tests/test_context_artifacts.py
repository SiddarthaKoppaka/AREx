"""Derived artifacts and compact prompt frames preserve exact evidence."""

from datetime import UTC, datetime

import pytest

from arc_agi_3.adapters.observation_projection import observation_view
from arc_agi_3.cognition.scratchpad import ScratchpadStore
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.scratchpad import ScratchpadUpdates, VerifiedFact
from arc_agi_3.trace.artifacts import EventArtifactStore
from arc_agi_3.trace.store import JsonlEventStore


def test_large_event_artifact_is_local_and_content_verified(tmp_path) -> None:
    events = JsonlEventStore(
        tmp_path / "events.jsonl",
        "run",
        "episode",
        "main",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    event = events.append(
        EventType.OBSERVATION, "environment", 0, {"frame": [1] * 5000}
    )
    artifacts = EventArtifactStore(tmp_path / "artifacts")
    reference = artifacts.archive(event)
    assert reference is not None
    assert artifacts.read(reference) == event.payload
    assert events.read()[0].payload == event.payload
    target = tmp_path / reference
    target.write_text('{"frame":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        artifacts.read(reference)


def test_current_scene_uses_lossless_rle_when_smaller() -> None:
    observation = Observation.build(
        game_id="scene",
        frame=[[[2] * 64 for _ in range(64)], [[3] * 64 for _ in range(64)]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1, 6],
    )
    projected = observation_view(observation)
    frame_view = projected["frame_view"]
    assert isinstance(frame_view, dict)
    assert frame_view["encoding"] == "row_rle_v1"
    assert len(frame_view["layers"]) == 2
    assert projected["observation_hash"] == observation.observation_hash


def test_scratchpad_consolidation_keeps_all_evidence() -> None:
    store = ScratchpadStore(token_budget=2048)
    initial = ScratchpadUpdates(
        add_verified_fact=(
            VerifiedFact(
                fact_id="first",
                version=1,
                fact="A",
                confidence=0.8,
                evidence_refs=("event-1",),
            ),
            VerifiedFact(
                fact_id="second",
                version=1,
                fact="B",
                confidence=0.8,
                evidence_refs=("event-2",),
            ),
        )
    )
    store.apply(initial, {"event-1", "event-2"})
    merged = VerifiedFact(
        fact_id="combined",
        version=1,
        fact="A and B",
        confidence=0.8,
        evidence_refs=("event-1", "event-2"),
        supersedes=("first", "second"),
    )
    store.apply(ScratchpadUpdates(add_verified_fact=(merged,)), {"event-1", "event-2"})
    assert [fact.fact_id for fact in store.current.verified_facts] == ["combined"]


def test_scratchpad_rejects_duplicate_superseded_ids() -> None:
    store = ScratchpadStore(token_budget=2048)
    first = VerifiedFact(
        fact_id="first", version=1, fact="A", confidence=0.8, evidence_refs=("event-1",)
    )
    store.apply(ScratchpadUpdates(add_verified_fact=(first,)), {"event-1"})
    duplicate = first.model_copy(
        update={"fact_id": "merged", "supersedes": ("first", "first")}
    )
    with pytest.raises(ValueError, match="unique"):
        store.apply(ScratchpadUpdates(add_verified_fact=(duplicate,)), {"event-1"})
