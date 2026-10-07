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


def failure_turn(io: EpisodeIO, current_turn: int) -> int:
    """The turn in progress when a crash occurs may be ahead of the
    session's own counter, which only advances after a successful return;
    events for that turn were already recorded before the crash."""
    history = io.events.read()
    last_step = max((event.step_id for event in history), default=current_turn)
    return max(current_turn, last_step)


def record_failure(
    io: EpisodeIO, turn: int, error: Exception, *, action_pending: bool
) -> None:
    """A first-class recoverable event, not a silent crash: what to retry, what
    is already safe (the last checkpoint), and what is still unverified."""
    history = io.events.read()
    payload: dict[str, JsonValue] = {
        "turn": turn,
        "error_type": type(error).__name__,
        "message": str(error),
        "environment_action_pending": action_pending,
        "checkpoint_available": any(
            event.event_type is EventType.CHECKPOINT for event in history
        ),
        "last_decision_event_id": _last_decision_event_id(history),
    }
    details = getattr(error, "trace_payload", None)
    if isinstance(details, dict):
        attempts = details.get("attempts")
        payload["error_categories"] = _categories(attempts)
        payload["details"] = persist_debug_previews(details, io.events.path.parent)
    io.append(EventType.FAILURE, "runtime", turn, payload)


def _last_decision_event_id(history: list[EventEnvelope]) -> str | None:
    for event in reversed(history):
        if event.event_type is EventType.MODEL_DECISION:
            return event.event_id
    return None


def _categories(attempts: object) -> list[JsonValue]:
    if not isinstance(attempts, list):
        return []
    seen: list[JsonValue] = []
    for item in attempts:
        category = item.get("error_category") if isinstance(item, dict) else None
        if isinstance(category, str) and category not in seen:
            seen.append(category)
    return seen


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
