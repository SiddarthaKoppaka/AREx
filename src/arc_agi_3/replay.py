"""Deterministic trace integrity and metric replay."""

from pathlib import Path

from arc_agi_3.config import EvaluatorConfig
from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.evaluation import EvaluationMetrics
from arc_agi_3.evaluation import evaluate
from arc_agi_3.replay_adapters import RecordedEnvironment, RecordedModel
from arc_agi_3.trace.store import JsonlEventStore


class ReplayResult(Contract):
    event_count: int
    last_event_hash: str
    metrics: EvaluationMetrics


def load_recorded_adapters(
    path: Path,
) -> tuple[RecordedEnvironment, RecordedModel]:
    store = JsonlEventStore(path, "fixture", "fixture", "fixture")
    events = store.read()
    return RecordedEnvironment(events), RecordedModel(events)


def replay_trace(path: Path, config: EvaluatorConfig) -> ReplayResult:
    store = JsonlEventStore(path, "replay", "replay", "replay")
    events = store.read()
    if not store.last_hash:
        raise ValueError("cannot replay an empty trace")
    return ReplayResult(
        event_count=len(events),
        last_event_hash=store.last_hash,
        metrics=evaluate(events, config),
    )
