"""Composition root for an offline run."""

from collections.abc import Callable
from datetime import datetime

from arc_agi_3.adapters.protocols import EnvironmentAdapter, ModelAdapter
from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.config import RunConfig
from arc_agi_3.manifest import RunManifest, build_manifest, write_manifest
from arc_agi_3.trace.checkpoints import CheckpointStore
from arc_agi_3.trace.store import JsonlEventStore

from .runner import EpisodeRunner


def build_runner(
    config: RunConfig,
    environment: EnvironmentAdapter,
    model: ModelAdapter,
    *,
    clock: Callable[[], datetime] | None = None,
    id_factory: Callable[[], str] | None = None,
    manifest: RunManifest | None = None,
) -> EpisodeRunner:
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
    )
    return EpisodeRunner(
        config=config,
        environment=environment,
        model=model,
        events=events,
        checkpoints=CheckpointStore(run_dir / "checkpoints"),
        ledger=BudgetLedger(config.budget.limits),
        manifest=resolved_manifest,
    )
