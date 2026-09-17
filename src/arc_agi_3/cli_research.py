"""Research and offline deployment commands."""

from pathlib import Path
from typing import Annotated

import typer

from arc_agi_3.deployment.hardware import detect_hardware
from arc_agi_3.deployment.kaggle import offline_smoke
from arc_agi_3.research.export import export_rows, run_row
from arc_agi_3.research.index import TraceIndex
from arc_agi_3.trace.canonical import canonical_json

app = typer.Typer(help="Derived research artifacts and offline checks.")


@app.command("index-trace")
def index_trace(
    trace: Path,
    output: Annotated[Path, typer.Option(help="Derived SQLite path.")],
) -> None:
    """Rebuild a disposable SQLite/FTS index from a verified trace."""
    count = TraceIndex(output).rebuild(trace)
    typer.echo(f"indexed events={count} path={output}")


@app.command("export-run")
def export_run(run_dir: Path, output: Path) -> None:
    """Export one run as JSONL, CSV, or optional Parquet."""
    export_rows([run_row(run_dir)], output)
    typer.echo(f"exported path={output}")


@app.command("hardware")
def hardware() -> None:
    """Print the runtime hardware record for experiment provenance."""
    typer.echo(canonical_json(detect_hardware()))


@app.command("offline-smoke")
def smoke(
    output: Annotated[Path, typer.Option(help="Artifact root.")] = Path("runs"),
) -> None:
    """Run the dependency-free Kaggle entrypoint with deterministic adapters."""
    typer.echo(canonical_json(offline_smoke(output)))
