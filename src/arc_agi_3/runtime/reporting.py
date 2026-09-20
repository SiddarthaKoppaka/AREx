"""Live public run reporting without changing the canonical event trace."""

import json
from pathlib import Path
from typing import Literal

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.trace.canonical import canonical_json

from .reporting_fields import PUBLIC_FIELDS

TraceMode = Literal["silent", "readable", "json"]


class LiveReporter:
    def __init__(
        self, mode: TraceMode = "silent", log_path: Path | None = None
    ) -> None:
        self.mode = mode
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, line: str) -> None:
        if self.mode == "silent":
            return
        print(line, flush=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def stage(self, name: str, **details: object) -> None:
        if self.mode == "json":
            self._write(json.dumps({"stage": name, **details}, default=str))
        else:
            summary = " ".join(f"{key}={value}" for key, value in details.items())
            self._write(f"[{name}] {summary}".rstrip())

    def on_event(self, event: EventEnvelope) -> None:
        if self.mode == "silent":
            return
        if self.mode == "json":
            self._write(canonical_json(event))
            return
        payload = event.payload
        details: dict[str, object] = {
            key: payload[key] for key in PUBLIC_FIELDS if key in payload
        }
        decision = payload.get("decision")
        if isinstance(decision, dict):
            details.update(
                {key: decision[key] for key in PUBLIC_FIELDS if key in decision}
            )
        usage = payload.get("usage")
        if isinstance(usage, dict):
            details["usage"] = usage
        failure = payload.get("details")
        if isinstance(failure, dict):
            details["generation_attempts"] = failure.get("generation_attempts")
            details["usage"] = failure.get("usage")
        if event.event_type.value == "evaluation":
            details = dict(payload)
        if event.event_type.value == "observation":
            frame = event.payload.get("frame")
            if isinstance(frame, list):
                layer = frame[0] if frame and isinstance(frame[0], list) else []
                row = layer[0] if layer and isinstance(layer[0], list) else []
                details["frame_dimensions"] = (len(frame), len(layer), len(row))
        if event.budget:
            details["budget"] = event.budget
        self._write(
            f"[{event.run_id} turn={event.step_id} {event.event_type.value}] "
            + json.dumps(details, default=str, separators=(",", ":"))[:1200]
        )
