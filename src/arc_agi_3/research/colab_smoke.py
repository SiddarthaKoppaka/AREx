"""Reusable composition for the public-game Colab integration smoke."""

from importlib import import_module
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from arc_agi_3.adapters.transformers_loader import QuantizationMode
from arc_agi_3.contracts.base import Contract


class ColabSmokeConfig(Contract):
    dataset_root: Path = Path("/content/arc_data/environment_files")
    game_id: str = "ls20"
    seed: int = 0
    model: str = "Qwen/Qwen3-8B"
    revision: str = "main"
    output_dir: Path = Path("/content/AREx/runs")
    run_id: str = "ls20-qwen3-8b-smoke-001"
    max_actions: int = Field(default=12, gt=0)
    max_model_calls: int = Field(default=4, gt=0)
    max_input_tokens: int = Field(default=100_000, gt=0)
    max_output_tokens: int = Field(default=8_000, gt=0)
    max_new_tokens: int = Field(default=1_024, gt=0)
    max_repairs: int = Field(default=1, ge=0)
    quantization: Literal["auto", "none", "nf4"] = "auto"


def resolve_quantization(requested: str, vram_gib: float | None) -> QuantizationMode:
    if requested != "auto":
        if requested not in ("none", "nf4"):
            raise ValueError(f"unsupported quantization: {requested}")
        return cast(QuantizationMode, requested)
    if vram_gib is None:
        raise RuntimeError("auto quantization requires a CUDA GPU")
    return "none" if vram_gib >= 24 else "nf4"


def gpu_vram_gib() -> float | None:
    torch = import_module("torch")
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.get_device_properties(0).total_memory / 1024**3)


def run_colab_smoke(settings: ColabSmokeConfig) -> dict[str, object]:
    from .colab_runtime import execute_smoke

    return execute_smoke(settings)
