"""Python 3.12 entrypoint for the maintained Colab notebook."""

import argparse
import json
from pathlib import Path

from arc_agi_3.settings import resolve_runtime
from arc_agi_3.trace.canonical import canonical_json

from .colab_experiment import run_colab_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="colab_transformers")
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    overrides = json.loads(options.overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("overrides-json must be a JSON object")
    settings = resolve_runtime(options.profile, overrides=overrides)
    result = run_colab_experiment(
        settings, options.dataset_root, game_id=options.game_id, seed=options.seed
    )
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
