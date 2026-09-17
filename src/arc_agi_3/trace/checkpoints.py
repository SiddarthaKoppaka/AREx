"""Atomic checkpoint snapshots linked to the immutable trace."""

import os
from pathlib import Path

from arc_agi_3.contracts.events import CheckpointRecord

from .canonical import canonical_hash, canonical_json


class CheckpointStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: CheckpointRecord) -> Path:
        expected = canonical_hash(checkpoint.model_dump(exclude={"checksum"}))
        if checkpoint.checksum != expected:
            raise ValueError("checkpoint checksum does not match content")
        target = self.directory / f"{checkpoint.checkpoint_id}.json"
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(canonical_json(checkpoint))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        return target

    def load(self, checkpoint_id: str) -> CheckpointRecord:
        if Path(checkpoint_id).name != checkpoint_id:
            raise ValueError("checkpoint ID must not contain a path")
        path = self.directory / f"{checkpoint_id}.json"
        checkpoint = CheckpointRecord.model_validate_json(path.read_text())
        expected = canonical_hash(checkpoint.model_dump(exclude={"checksum"}))
        if checkpoint.checksum != expected:
            raise ValueError("checkpoint checksum does not match content")
        return checkpoint
