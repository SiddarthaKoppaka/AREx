"""Composition root for an offline run."""

from collections.abc import Callable
from datetime import datetime

from arc_agi_3.adapters.protocols import EnvironmentAdapter, ModelAdapter
from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.config import RunConfig
from arc_agi_3.manifest import RunManifest, build_manifest, write_manifest
from arc_agi_3.trace.checkpoints import CheckpointStore
from arc_agi_3.trace.store import JsonlEventStore

from .io import EpisodeIO
from .reporting import LiveReporter
from .runner import EpisodeRunner
from .session import CognitiveSession


def build_session(
    config: RunConfig,
    environment: EnvironmentAdapter,
    model: ModelAdapter,
    *,
    clock: Callable[[], datetime] | None = None,
    id_factory: Callable[[], str] | None = None,
    manifest: RunManifest | None = None,
    reporter: LiveReporter | None = None,
) -> CognitiveSession:
    run_dir = config.output_dir / config.run_id
    event_path = run_dir / "events.jsonl"
    if event_path.exists():
        raise FileExistsError(f"run trace already exists: {event_path}")
    resolved_manifest = manifest or build_manifest(
        config, model.metadata, environment.metadata
    )
    write_manifest(run_dir / "manifest.json", resolved_manifest)
    events = JsonlEventStore(
        event_path,
        config.run_id,
        config.episode_id,
        config.branch_id,
        clock=clock,
        id_factory=id_factory,
        observers=(reporter.on_event,) if reporter else (),
    )
    io = EpisodeIO(
        config,
        environment,
        events,
        CheckpointStore(run_dir / "checkpoints"),
        BudgetLedger(config.budget.limits),
    )
    return CognitiveSession(io, model, resolved_manifest)


def build_runner(
    config: RunConfig,
    environment: EnvironmentAdapter,
    model: ModelAdapter,
    *,
    clock: Callable[[], datetime] | None = None,
    id_factory: Callable[[], str] | None = None,
    manifest: RunManifest | None = None,
    reporter: LiveReporter | None = None,
    raise_on_failure: bool = False,
) -> EpisodeRunner:
    session = build_session(
        config,
        environment,
        model,
        clock=clock,
        id_factory=id_factory,
        manifest=manifest,
        reporter=reporter,
    )
    return EpisodeRunner(
        environment=environment, session=session, raise_on_failure=raise_on_failure
    )
