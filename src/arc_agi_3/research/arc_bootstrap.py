"""Explicitly download missing public ARC assets before an offline run."""

import argparse
from importlib import import_module
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    arc_agi = import_module("arc_agi")
    options.dataset_root.mkdir(parents=True, exist_ok=True)
    offline = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(options.dataset_root),
    )
    if (
        offline.make(options.game_id, seed=options.seed, include_frame_data=True)
        is not None
    ):
        print("ARC environment already available offline", flush=True)
        return
    online = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.NORMAL,
        environments_dir=str(options.dataset_root),
    )
    if online.make(options.game_id, seed=options.seed, include_frame_data=True) is None:
        raise RuntimeError("ARC bootstrap failed; check ARC_API_KEY and network")
    print("ARC environment downloaded for offline execution", flush=True)


if __name__ == "__main__":
    main()
