"""Lossless mechanical views and explicit retrieval budgets."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from arc_agi_3.context.evidence_retrieval import retrieve_evidence
from arc_agi_3.context.frame_views import decode_rle_frame, frame_summary, rle_frame
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.evidence import EvidenceQuery, FrameRegion
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.trace.store import JsonlEventStore


def _events(tmp_path: Path) -> list[EventEnvelope]:
    store = JsonlEventStore(
        tmp_path / "events.jsonl",
        "run",
        "episode",
        "main",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    for step, value in enumerate((1, 2)):
        observation = Observation.build(
            game_id="test",
            frame=[[[1, value], [1, 3]], [[4, 4], [5, 4]]],
            state=EnvironmentState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            available_actions=[1],
        )
        store.append(
            EventType.OBSERVATION,
            "environment",
            step,
            observation.model_dump(mode="json"),
        )
    return store.read()


def test_rle_round_trip_and_mechanical_summary() -> None:
    frame = [[[1, 1, 2], [2, 2, 2]], [[4], [4]]]
    encoded = rle_frame(frame)
    assert decode_rle_frame(encoded) == frame
    summary = frame_summary(frame)
    assert summary["value_histogram"] == {"1": 2, "2": 4, "4": 2}
    assert summary["component_count"] == 3


def test_rle_preserves_ragged_rows_and_rejects_invalid_runs() -> None:
    frame = [[[1, 1], [], [2]], [], [[7, 8, 8]]]
    assert decode_rle_frame(rle_frame(frame)) == frame
    with pytest.raises(ValueError, match="invalid RLE run"):
        decode_rle_frame({"encoding": "row_rle_v1", "layers": [[[[3, 0]]]]})


def test_transition_delta_region_and_full_views(tmp_path: Path) -> None:
    history = _events(tmp_path)
    second = history[-1]
    base = EvidenceQuery(
        event_ids=(second.event_id,),
        purpose="compare exact cells",
        token_budget=3000,
        view="transition_delta",
    )
    _, delta, refs = retrieve_evidence(history, base)
    assert refs == (second.event_id,)
    assert delta["items"][0]["content"]["changes"] == [[0, 0, 1, 1, 2]]
    _, region, _ = retrieve_evidence(
        history,
        base.model_copy(
            update={
                "view": "region",
                "region": FrameRegion(layer=1, x=0, y=0, width=2, height=1),
            }
        ),
    )
    assert region["items"][0]["content"]["cells"] == [[4, 4]]
    _, full, _ = retrieve_evidence(history, base.model_copy(update={"view": "full"}))
    assert full["items"][0]["content"] == second.payload


def test_small_budget_explicitly_omits_payload(tmp_path: Path) -> None:
    history = _events(tmp_path)
    status, result, refs = retrieve_evidence(
        history,
        EvidenceQuery(
            event_ids=(history[0].event_id,),
            view="full",
            purpose="inspect original frame",
            token_budget=256,
        ),
    )
    assert status == "partial"
    assert refs == ()
    assert result["omitted_event_ids"] == [history[0].event_id]
    assert Observation.model_validate(history[0].payload).frame[0][0] == (1, 1)
