"""Offline content-addressed copies of large canonical event payloads."""

import os
from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.contracts.events import EventEnvelope

from .canonical import canonical_hash, canonical_json


class EventArtifactStore:
    def __init__(self, directory: Path, minimum_bytes: int = 4096) -> None:
        self.directory = directory
        self.minimum_bytes = minimum_bytes
        self.directory.mkdir(parents=True, exist_ok=True)

    def archive(self, event: EventEnvelope) -> str | None:
        serialized = canonical_json(event.payload)
        if len(serialized.encode("utf-8")) < self.minimum_bytes:
            return None
        return self._write(event.payload, serialized)

    def archive_many(self, events: list[EventEnvelope]) -> None:
        for event in events:
            self.archive(event)

    def reference(self, event: EventEnvelope) -> str | None:
        if len(canonical_json(event.payload).encode("utf-8")) < self.minimum_bytes:
            return None
        reference = f"artifacts/{canonical_hash(event.payload)}.json"
        return reference if (self.directory / Path(reference).name).is_file() else None

    def archive_document(self, value: dict[str, JsonValue]) -> str:
        serialized = canonical_json(value)
        if len(serialized.encode("utf-8")) > 65_536:
            raise ValueError("artifact document exceeds 65536 bytes")
        return self._write(value, serialized)

    def _write(self, value: dict[str, JsonValue], serialized: str) -> str:
        digest = canonical_hash(value)
        target = self.directory / f"{digest}.json"
        if not target.exists():
            temporary = self.directory / f".{digest}.{os.getpid()}.tmp"
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(target)
        return f"artifacts/{digest}.json"

    def read(self, reference: str) -> dict[str, JsonValue]:
        path = Path(reference)
        if (
            not reference.startswith("artifacts/")
            or path.parts != ("artifacts", path.name)
            or path.suffix != ".json"
        ):
            raise ValueError("artifact reference must be a content hash")
        digest = path.stem
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact reference must be a SHA-256 hash")
        from json import loads

        value = loads((self.directory / path.name).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or canonical_hash(value) != digest:
            raise ValueError("artifact content hash mismatch")
        return value
