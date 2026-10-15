"""Research and offline deployment commands."""

from pathlib import Path
from typing import Annotated

import typer

from arc_agi_3.deployment.hardware import detect_hardware
from arc_agi_3.deployment.kaggle import offline_smoke
from arc_agi_3.research.baseline_report import build_report
from arc_agi_3.research.export import export_rows, run_row
from arc_agi_3.research.index import TraceIndex
from arc_agi_3.research.report_comparison import compare_reports
from arc_agi_3.research.report_markdown import render_markdown
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


@app.command("compare-runs")
def compare_runs(
    control: Path,
    candidate: Path,
    output: Annotated[Path, typer.Option(help="JSON comparison report path.")],
    markdown: Annotated[
        Path | None, typer.Option(help="Optional Markdown report path.")
    ] = None,
) -> None:
    """Compare two run directories; write a JSON report and optional Markdown."""
    comparison = compare_reports(build_report(control), build_report(candidate))
    output.write_text(canonical_json(comparison) + "\n")
    if markdown is not None:
        markdown.write_text(render_markdown(comparison))
    typer.echo(f"compared output={output}")


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
