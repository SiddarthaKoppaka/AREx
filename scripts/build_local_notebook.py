"""Generate a local Transformers notebook from the shared runtime profile."""

import json
from pathlib import Path

try:
    from scripts.build_colab_notebook import cell
except ModuleNotFoundError:
    from build_colab_notebook import cell

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "arc_agi_3_local_transformers.ipynb"


def build() -> dict[str, object]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            cell(
                "markdown",
                "intro",
                """
                # AREx local Transformers run
                Install with `uv sync --extra colab`, then select a local model
                directory and offline ARC environment root in the next cell.
            """,
            ),
            cell(
                "code",
                "run-local",
                """
                import json, subprocess, sys
                from pathlib import Path
                OVERRIDES = {
                    'model_path': '/path/to/local/model',
                    'model_name': 'YOUR_MODEL_ID',
                    'output_dir': str(Path.cwd() / 'runs'),
                    'log_dir': str(Path.cwd() / 'runs' / 'logs'),
                }
                DATASET_ROOT = '/path/to/offline/environment_files'
                assert 'YOUR_MODEL_ID' not in OVERRIDES['model_name']
                assert '/path/to/' not in OVERRIDES['model_path']
                assert '/path/to/' not in DATASET_ROOT
                subprocess.run([
                    sys.executable, '-m', 'arc_agi_3.research.colab_entry',
                    '--profile', 'local_smoke',
                    '--overrides-json', json.dumps(OVERRIDES),
                    '--dataset-root', DATASET_ROOT,
                    '--game-id', 'ls20', '--seed', '0',
                ], check=True)
            """,
            ),
        ],
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n")


if __name__ == "__main__":
    main()
