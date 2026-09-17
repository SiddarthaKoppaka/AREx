#!/usr/bin/env bash
set -euo pipefail

wheelhouse="${1:-dist/kaggle-wheelhouse}"
python_version="${PYTHON_VERSION:-3.12}"
platform="${KAGGLE_PLATFORM:-manylinux_2_28_x86_64}"

mkdir -p "$wheelhouse"
uv build --wheel --out-dir "$wheelhouse"
uv export --extra kaggle-inference --no-dev --no-emit-project \
  --no-emit-package torch \
  --format requirements-txt --output-file "$wheelhouse/requirements.txt"

# Torch deliberately comes from Kaggle's CUDA image; replacing it can break CUDA.
uv run --with pip python -m pip download \
  --dest "$wheelhouse" \
  --requirement "$wheelhouse/requirements.txt" \
  --platform "$platform" \
  --python-version "$python_version" \
  --implementation cp \
  --only-binary=:all: