"""Checkpoint checksums and atomic round trips."""

from pathlib import Path

import pytest

from arc_agi_3.contracts.events import CheckpointRecord
from arc_agi_3.trace.canonical import canonical_hash
from arc_agi_3.trace.checkpoints import CheckpointStore


def checkpoint() -> CheckpointRecord:
    draft = CheckpointRecord(
        checkpoint_id="checkpoint-1",
        run_id="run",
        episode_id="episode",
        branch_id="main",
        step_id=1,
        last_event_hash="event-hash",
        observation={"state": "WIN"},
        budget={"consumed": {"actions": 1}},
        environment_state={"position": 2},
        checksum="",
    )
    digest = canonical_hash(draft.model_dump(exclude={"checksum"}))
    return draft.model_copy(update={"checksum": digest})


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    expected = checkpoint()
    store.save(expected)
    assert store.load(expected.checkpoint_id) == expected


def test_bad_checksum_is_rejected(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    with pytest.raises(ValueError):
        store.save(checkpoint().model_copy(update={"checksum": "bad"}))
