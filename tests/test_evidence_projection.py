"""Selected evidence views survive model-facing event projection."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from arc_agi_3.context.evidence_retrieval import retrieve_evidence
from arc_agi_3.context.projection import compact_context_event
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.evidence import EvidenceQuery
from arc_agi_3.trace.store import JsonlEventStore


def test_rle_tool_result_stays_visible(tmp_path: Path) -> None:
    store = JsonlEventStore(
        tmp_path / "events.jsonl",
        "run",
        "episode",
        "main",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    source = store.append(
        EventType.OBSERVATION,
        "environment",
        0,
        {"frame": [[[1, 1, 2], [3, 3, 3]]], "observation_hash": "hash"},
    )
    status, output, refs = retrieve_evidence(
        store.read(),
        EvidenceQuery(
            event_ids=(source.event_id,),
            view="rle_frame",
            purpose="inspect exact row runs",
            token_budget=1200,
        ),
    )
    assert status == "complete"
    assert refs == (source.event_id,)
    recorded = store.append(
        EventType.TOOL_RESULT,
        "tools",
        0,
        {"result": {"status": status, "output": {"evidence": output}}},
    )
    projected = compact_context_event(recorded)
    visible = cast(Any, projected.payload["result"])
    assert visible["evidence"]["items"][0]["content"]["layers"] == [
        [[[1, 2], [2, 1]], [[3, 3]]]
    ]
    assert store.read()[0].payload["frame"] == source.payload["frame"]


def test_metadata_budget_failure_is_explicit(tmp_path: Path) -> None:
    store = JsonlEventStore(
        tmp_path / "events.jsonl",
        "run",
        "episode",
        "main",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    source = store.append(EventType.OBSERVATION, "environment", 0, {"frame": [[[1]]]})
    with pytest.raises(ValueError, match="metadata exceeds"):
        retrieve_evidence(
            store.read(),
            EvidenceQuery(
                event_ids=(source.event_id,),
                view="full",
                purpose="exact payload",
                token_budget=128,
            ),
        )
