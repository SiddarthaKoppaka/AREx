"""Generate the official Kaggle notebook and kernel metadata from one config."""

import json
from pathlib import Path

from arc_agi_3.settings import resolve_runtime

try:
    from scripts.submission_cells import build
except ModuleNotFoundError:
    from submission_cells import build

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "kaggle" / "config.json"
AGENT = ROOT / "kaggle" / "agent" / "my_agent.py"
NOTEBOOK = ROOT / "kaggle" / "submission.ipynb"
METADATA = ROOT / "kaggle" / "kernel-metadata.json"


def resolved_config() -> dict[str, object]:
    config: dict[str, object] = json.loads(CONFIG.read_text())
    kernel = config["kernel"]
    assert isinstance(kernel, dict)
    source = kernel["model_sources"][0]
    _, slug, framework, variation, version = source.split("/")
    mount = f"/kaggle/input/{slug}/{framework}/{variation}/{version}"
    profile = resolve_runtime("kaggle_submission", environ={})
    if str(profile.model_path) != mount:
        raise ValueError("Kaggle profile model_path differs from attached model source")
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    runtime.update(profile=profile.profile, model_path=mount)
    return config


def main() -> None:
    config = resolved_config()
    kernel = config["kernel"]
    assert isinstance(kernel, dict)
    notebook = build(config, AGENT.read_text())
    metadata = {
        "id": kernel["id"],
        "title": kernel["title"],
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": kernel["competition_sources"],
        "dataset_sources": kernel["dataset_sources"],
        "model_sources": kernel["model_sources"],
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n")
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
