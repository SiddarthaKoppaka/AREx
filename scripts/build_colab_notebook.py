"""Generate the maintained A100/Transformers Colab notebook."""

import json
from pathlib import Path
from textwrap import dedent

try:
    from scripts.colab_setup_cells import setup_cells
except ModuleNotFoundError:
    from colab_setup_cells import setup_cells

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "arc_agi_3_colab_transformers.ipynb"


def cell(kind: str, cell_id: str, source: str) -> dict[str, object]:
    item: dict[str, object] = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build() -> dict[str, object]:
    cells = [cell(*item) for item in setup_cells()]
    cells.extend(
        [
            cell(
                "code",
                "prepare-arc-assets",
                """
            import subprocess
            DATASET_ROOT = '/content/drive/MyDrive/AREx/environment_files'
            subprocess.run([
                '/content/arex-py312/bin/python', '-m',
                'arc_agi_3.research.arc_bootstrap',
                '--dataset-root', DATASET_ROOT, '--game-id', 'ls20', '--seed', '0',
            ], check=True)
        """,
            ),
            cell(
                "code",
                "run-experiment",
                """
            import json, subprocess
            from pathlib import Path
            import torch
            assert torch.cuda.is_available(), 'A CUDA GPU is required'
            gpu = torch.cuda.get_device_properties(0)
            print(f'GPU: {gpu.name}; VRAM: {gpu.total_memory / 1024**3:.1f} GiB')
            OVERRIDES = {
                'model_path': (
                    '/content/drive/MyDrive/AREx/models/'
                    'REPLACE_WITH_MODEL_DIRECTORY'),
                'model_name': 'REPLACE_WITH_MODEL_ID',
            }
            assert 'REPLACE_WITH' not in json.dumps(OVERRIDES)
            subprocess.run([
                '/content/arex-py312/bin/python', '-m',
                'arc_agi_3.research.colab_entry',
                '--profile', 'colab_transformers',
                '--overrides-json', json.dumps(OVERRIDES),
                '--dataset-root', '/content/drive/MyDrive/AREx/environment_files',
                '--game-id', 'ls20', '--seed', '0',
            ], check=True)
        """,
            ),
            cell(
                "markdown",
                "artifacts",
                """
            Each run gets a unique ID under `/content/drive/MyDrive/AREx/runs`.
            Inspect `events.jsonl`, `manifest.json`, `checkpoints/`, and `result.json`.
            The console log is under `/content/drive/MyDrive/AREx/logs`.
        """,
            ),
        ]
    )
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n")


if __name__ == "__main__":
    main()
