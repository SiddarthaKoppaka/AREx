#!/usr/bin/env bash
set -euo pipefail

wheelhouse="${1:-dist/wheelhouse}"
mkdir -p "$wheelhouse"
uv build --wheel --out-dir "$wheelhouse"
uv export --no-dev --no-emit-project --format requirements-txt \
  --output-file "$wheelhouse/requirements.txt"
uv run --with pip python -m pip download \
  --dest "$wheelhouse" -r "$wheelhouse/requirements.txt"
