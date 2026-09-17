"""Atomically rebuildable SQLite/FTS5 index over immutable events."""

import os
import sqlite3
from pathlib import Path

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.trace.canonical import canonical_json
from arc_agi_3.trace.store import JsonlEventStore

SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE events(sequence INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL,
 event_type TEXT NOT NULL, branch_id TEXT NOT NULL, step_id INTEGER NOT NULL,
 component TEXT NOT NULL, payload_json TEXT NOT NULL, event_hash TEXT NOT NULL);
CREATE VIRTUAL TABLE event_fts USING fts5(event_id, event_type, component, payload);
"""


class TraceIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def rebuild(self, trace: Path) -> int:
        store = JsonlEventStore(trace, "index", "index", "index")
        events = store.read()
        temporary = self.path.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(temporary) as connection:
            connection.executescript(SCHEMA)
            self._insert(connection, trace, events)
        os.replace(temporary, self.path)
        return len(events)

    @staticmethod
    def _insert(
        connection: sqlite3.Connection, trace: Path, events: list[EventEnvelope]
    ) -> None:
        rows = [
            (
                e.sequence,
                e.event_id,
                str(e.event_type),
                e.branch_id,
                e.step_id,
                e.component,
                canonical_json(e.payload),
                e.event_hash,
            )
            for e in events
        ]
        connection.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)", rows)
        connection.executemany(
            "INSERT INTO event_fts VALUES(?,?,?,?)",
            [(row[1], row[2], row[5], row[6]) for row in rows],
        )
        metadata = {
            "source_trace": str(trace),
            "event_count": str(len(events)),
            "last_event_hash": events[-1].event_hash if events else "",
        }
        connection.executemany("INSERT INTO metadata VALUES(?,?)", metadata.items())

    def search(self, query: str, limit: int = 20) -> list[tuple[str, str]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT event_id, payload FROM event_fts "
                "WHERE event_fts MATCH ? LIMIT ?",
                (query, limit),
            ).fetchall()
        return [(str(event_id), str(payload)) for event_id, payload in rows]
