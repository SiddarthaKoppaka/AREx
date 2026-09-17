"""Thin local entrypoints; production adapters remain optional."""

from pathlib import Path
from typing import Annotated

import typer

from arc_agi_3.cli_research import app as research_app
from arc_agi_3.config import RunConfig
from arc_agi_3.runtime.factory import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script
from arc_agi_3.trace.canonical import canonical_json
from arc_agi_3.trace.store import JsonlEventStore

app = typer.Typer(no_args_is_help=True)
app.add_typer(research_app, name="research")


@app.command("fake-run")
def fake_run(
    output: Annotated[Path, typer.Option(help="Artifact root.")] = Path("runs"),
    run_id: Annotated[str, typer.Option(help="Stable run identifier.")] = "fake-run",
) -> None:
    """Run the deterministic offline foundation slice."""
    trace = output / run_id / "events.jsonl"
    if trace.exists():
        raise typer.BadParameter(f"trace already exists: {trace}")
    config = RunConfig(
        run_id=run_id,
        experiment_id="foundation-smoke",
        output_dir=output,
        game_id="fake-line",
    )
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    typer.echo(canonical_json(runner.run()))


@app.command("verify-trace")
def verify_trace(path: Path) -> None:
    """Validate sequence numbers and the complete event hash chain."""
    store = JsonlEventStore(path, "verify", "verify", "verify")
    events = store.read()
    typer.echo(f"valid events={len(events)} last_hash={store.last_hash}")
