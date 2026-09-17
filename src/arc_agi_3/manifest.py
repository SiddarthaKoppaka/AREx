"""Run provenance captured before cognition begins."""

import platform
import subprocess
from pathlib import Path

from pydantic import JsonValue

from arc_agi_3 import __version__
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.base import Contract
from arc_agi_3.trace.canonical import canonical_json


class RunManifest(Contract):
    run_id: str
    experiment_id: str
    config_hash: str
    config: dict[str, JsonValue]
    harness_version: str
    git_commit: str
    git_dirty: bool
    python_version: str
    platform: str
    model: dict[str, JsonValue]
    environment: dict[str, JsonValue]


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, check=False, text=True)
    return result.stdout.strip()


def build_manifest(
    config: RunConfig,
    model: dict[str, JsonValue],
    environment: dict[str, JsonValue],
) -> RunManifest:
    commit = _git_output("rev-parse", "HEAD") or "uncommitted"
    dirty = bool(_git_output("status", "--porcelain"))
    values = config.model_dump(mode="json", exclude={"output_dir"})
    return RunManifest(
        run_id=config.run_id,
        experiment_id=config.experiment_id,
        config_hash=config.config_hash,
        config=values,
        harness_version=__version__,
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        platform=platform.platform(),
        model=model,
        environment=environment,
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(manifest) + "\n")
    temporary.replace(path)
