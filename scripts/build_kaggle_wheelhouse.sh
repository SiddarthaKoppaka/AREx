#!/usr/bin/env bash
set -euo pipefail

wheelhouse="${1:-dist/kaggle-wheelhouse}"
python_version="${PYTHON_VERSION:-3.12}"
platform="${KAGGLE_PLATFORM:-manylinux2014_x86_64}"
omit_packages=(
  torch triton mpmath networkx sympy
  cuda-bindings cuda-pathfinder cuda-toolkit
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink
  nvidia-nvshmem-cu13 nvidia-nvtx
)
omit_args=()
for package in "${omit_packages[@]}"; do
  omit_args+=(--no-emit-package "$package")
done

mkdir -p "$wheelhouse"
find "$wheelhouse" -maxdepth 1 -type f -delete
uv build --wheel --out-dir "$wheelhouse"
uv export --extra kaggle-inference --no-dev --no-emit-project \
  "${omit_args[@]}" \
  --format requirements-txt --output-file "$wheelhouse/requirements.txt"

# Torch deliberately comes from Kaggle's CUDA image; replacing it can break CUDA.
uv run --with pip python -m pip download \
  --dest "$wheelhouse" \
  --requirement "$wheelhouse/requirements.txt" \
  --no-deps \
  --platform "$platform" \
  --python-version "$python_version" \
  --implementation cp \
  --only-binary=:all: