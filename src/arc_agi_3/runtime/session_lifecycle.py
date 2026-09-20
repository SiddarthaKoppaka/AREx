"""Lifecycle events and deterministic evaluation for cognitive sessions."""

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.evaluation import evaluate
from arc_agi_3.manifest import RunManifest

from .debug_output import persist_debug_previews
from .io import EpisodeIO
from .state import RunResult


def initialize_run(
    io: EpisodeIO, manifest: RunManifest, observation: Observation
) -> EventEnvelope:
    started = io.append(
        EventType.RUN_STARTED,
        "runtime",
        0,
        {"manifest": manifest.model_dump(mode="json")},
    )
    return io.append(
        EventType.OBSERVATION,
        "environment",
        0,
        observation.model_dump(mode="json"),
        (started.event_id,),
    )


def record_failure(io: EpisodeIO, turn: int, error: Exception) -> None:
    payload: dict[str, JsonValue] = {
        "error_type": type(error).__name__,
        "message": str(error),
    }
    details = getattr(error, "trace_payload", None)
    if isinstance(details, dict):
        payload["details"] = persist_debug_previews(details, io.events.path.parent)
    io.append(EventType.FAILURE, "runtime", turn, payload)


def finish_run(io: EpisodeIO, turn: int, reason: str) -> RunResult:
    io.append(EventType.RUN_FINISHED, "runtime", turn, {"reason": reason})
    metrics = evaluate(io.events.read(), io.config.evaluator)
    io.append(EventType.EVALUATION, "evaluator", turn, metrics.model_dump(mode="json"))
    return RunResult(
        run_id=io.config.run_id,
        stop_reason=reason,
        trace_path=str(io.events.path),
        metrics=metrics,
    )
