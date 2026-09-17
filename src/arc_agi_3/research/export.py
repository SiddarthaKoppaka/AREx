"""Flat, provenance-rich run exports for reproducible analysis."""

import csv
import json
from pathlib import Path

from arc_agi_3.config import EvaluatorConfig
from arc_agi_3.evaluation import evaluate
from arc_agi_3.manifest import RunManifest
from arc_agi_3.trace.canonical import canonical_json
from arc_agi_3.trace.store import JsonlEventStore


def run_row(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    trace_path = run_dir / "events.jsonl"
    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    events = JsonlEventStore(trace_path, "export", "export", "export").read()
    evaluator = EvaluatorConfig.model_validate(manifest.config.get("evaluator", {}))
    metrics = evaluate(events, evaluator)
    seed = manifest.config.get("seed", 0)
    return {
        "run_id": manifest.run_id,
        "experiment_id": manifest.experiment_id,
        "game_id": str(manifest.config.get("game_id", "")),
        "seed": seed if isinstance(seed, int) else 0,
        "config_hash": manifest.config_hash,
        "last_event_hash": events[-1].event_hash if events else "",
        "event_count": len(events),
        "model": canonical_json(manifest.model),
        "environment": canonical_json(manifest.environment),
        "ablations": canonical_json(manifest.config.get("ablations", {})),
        **metrics.model_dump(mode="json"),
    }


def export_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("at least one row is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        return
    if path.suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(_csv_rows(rows))
        return
    if path.suffix == ".parquet":
        _write_parquet(rows, path)
        return
    raise ValueError("export must be .jsonl, .csv, or .parquet")


def _csv_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, list | dict)
            else value
            for key, value in row.items()
        }
        for row in rows
    ]


def _write_parquet(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install the analytics extra for Parquet export") from error
    pq.write_table(pa.Table.from_pylist(_csv_rows(rows)), path)
