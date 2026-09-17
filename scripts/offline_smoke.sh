#!/usr/bin/env bash
set -euo pipefail

wheelhouse="${1:-dist/wheelhouse}"
smoke_dir="$(mktemp -d)"
trap 'rm -rf "$smoke_dir"' EXIT
uv venv --seed --python 3.12 "$smoke_dir/venv"
"$smoke_dir/venv/bin/python" -m pip install --no-index \
  --find-links "$wheelhouse" arc-agi-3
cd "$smoke_dir"
"$smoke_dir/venv/bin/arc-agi-3" fake-run --output runs --run-id offline
"$smoke_dir/venv/bin/arc-agi-3" verify-trace runs/offline/events.jsonl
