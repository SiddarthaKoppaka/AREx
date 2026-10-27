"""Rough VRAM fit check; run before loading so a bad model fails fast."""

from importlib import import_module


def weights_gb(params_billion: float, bits: int) -> float:
    return params_billion * bits / 8


def fits(
    params_billion: float,
    bits: int,
    vram_gb: float,
    *,
    kv_gb: float = 6.0,
    usable: float = 0.9,
) -> bool:
    """True if weights plus a KV-cache allowance fit in `usable` of the VRAM."""
    return weights_gb(params_billion, bits) + kv_gb <= vram_gb * usable


def preflight(params_billion: float | None, bits: int, vram_gb: float) -> None:
    """Raise before loading if the model clearly cannot fit (skips on no CUDA)."""
    if params_billion is None or vram_gb <= 0:
        return
    if not fits(params_billion, bits, vram_gb):
        need = weights_gb(params_billion, bits)
        raise RuntimeError(
            f"{params_billion}B @ {bits}-bit needs ~{need:.0f} GB weights + KV; "
            f"only {vram_gb:.0f} GB VRAM visible"
        )


def total_vram_gb() -> float:
    """Sum of visible CUDA devices' memory, or 0.0 without torch/CUDA."""
    try:
        torch = import_module("torch")
        count = torch.cuda.device_count()
        return float(
            sum(torch.cuda.get_device_properties(i).total_memory for i in range(count))
            / 1e9
        )
    except (ImportError, RuntimeError):
        return 0.0
