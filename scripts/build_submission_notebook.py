"""Generate the official Kaggle notebook and kernel metadata from one config."""

import json
from pathlib import Path

from scripts.submission_cells import build

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "kaggle" / "config.json"
AGENT = ROOT / "kaggle" / "agent" / "my_agent.py"
NOTEBOOK = ROOT / "kaggle" / "submission.ipynb"
METADATA = ROOT / "kaggle" / "kernel-metadata.json"


def main() -> None:
    config = json.loads(CONFIG.read_text())
    notebook = build(config, AGENT.read_text())
    kernel = config["kernel"]
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
