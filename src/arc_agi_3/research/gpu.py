"""Small CUDA capacity probe used by local and Colab workflows."""

from importlib import import_module


def gpu_vram_gib() -> float | None:
    torch = import_module("torch")
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.get_device_properties(0).total_memory / 1024**3)
