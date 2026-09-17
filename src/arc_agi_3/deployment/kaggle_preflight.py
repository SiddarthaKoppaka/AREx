"""Cheap competition checks performed before attached weights are loaded."""

from importlib import import_module
from shutil import disk_usage
from typing import Any
from uuid import uuid4

from .kaggle_settings import KaggleSettings


def run_preflight(settings: KaggleSettings) -> dict[str, Any]:
    failures: list[str] = []
    required = ("config.json", "tokenizer_config.json")
    for name in required:
        if not (settings.model_path / name).is_file():
            failures.append(f"missing model asset: {name}")
    if not tuple(settings.model_path.glob("*.safetensors")):
        failures.append("missing model asset: *.safetensors")
    try:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.output_dir / f".preflight-{uuid4().hex}"
        probe.write_text("ok")
        probe.unlink()
    except OSError:
        failures.append("output directory is not writable")
    for module in ("accelerate", "safetensors", "transformers"):
        try:
            import_module(module)
        except ImportError:
            failures.append(f"{module} is not installed")
    try:
        torch = import_module("torch")
        cuda = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        cuda, device = False, None
        failures.append("torch is not installed")
    if settings.require_gpu and not cuda:
        failures.append("CUDA GPU is required but unavailable")
    free_disk = 0
    if settings.output_dir.is_dir():
        free_disk = disk_usage(settings.output_dir).free
        if free_disk < 5 * 1024**3:
            failures.append("less than 5 GiB free in output directory")
    if failures:
        raise RuntimeError("Kaggle preflight failed: " + "; ".join(failures))
    return {"cuda": cuda, "device": device, "free_disk_bytes": free_disk}
