"""Crash-visible append-only JSONL event source of truth."""

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope

from .canonical import canonical_hash, canonical_json


class TraceIntegrityError(ValueError):
    pass


class JsonlEventStore:
    def __init__(
        self,
        path: Path,
        run_id: str,
        episode_id: str,
        branch_id: str,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.episode_id = episode_id
        self.branch_id = branch_id
        self.clock = clock or (lambda: datetime.now(UTC))
        self.id_factory = id_factory or (lambda: str(uuid4()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.read() if self.path.exists() else []
        self.sequence = len(existing)
        self.last_hash = existing[-1].event_hash if existing else None

    def append(
        self,
        event_type: EventType,
        component: str,
        step_id: int,
        payload: dict[str, JsonValue],
        *,
        causal_refs: tuple[str, ...] = (),
        budget: dict[str, JsonValue] | None = None,
    ) -> EventEnvelope:
        draft = EventEnvelope(
            event_id=self.id_factory(),
            run_id=self.run_id,
            episode_id=self.episode_id,
            branch_id=self.branch_id,
            step_id=step_id,
            sequence=self.sequence,
            timestamp=self.clock(),
            event_type=event_type,
            component=component,
            causal_refs=causal_refs,
            payload=payload,
            budget=budget or {},
            previous_event_hash=self.last_hash,
            event_hash="",
        )
        digest = canonical_hash(draft.model_dump(exclude={"event_hash"}))
        event = draft.model_copy(update={"event_hash": digest})
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1
        self.last_hash = event.event_hash
        return event

    def read(self) -> list[EventEnvelope]:
        events: list[EventEnvelope] = []
        previous: str | None = None
        if not self.path.exists():
            return events
        for sequence, line in enumerate(self.path.read_text().splitlines()):
            event = EventEnvelope.model_validate_json(line)
            values = event.model_dump(exclude={"event_hash"})
            if event.sequence != sequence or event.previous_event_hash != previous:
                raise TraceIntegrityError(f"broken event chain at sequence {sequence}")
            if canonical_hash(values) != event.event_hash:
                raise TraceIntegrityError(f"invalid event hash at sequence {sequence}")
            events.append(event)
            previous = event.event_hash
        return events
