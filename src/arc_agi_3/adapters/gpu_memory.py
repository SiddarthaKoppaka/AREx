"""Optional, best-effort CUDA memory telemetry and post-failure cleanup.

Torch is imported lazily so CPU-only test environments never need it
installed. Every function degrades to a no-op/None when torch or CUDA is
unavailable, so this module never changes behavior on CPU-only runs.
"""

from importlib import import_module
from typing import Any

from pydantic import JsonValue

from arc_agi_3.contracts.base import Contract


class GPUMemorySnapshot(Contract):
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0


def _cuda() -> Any | None:
    try:
        torch = import_module("torch")
    except ImportError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


def snapshot() -> GPUMemorySnapshot | None:
    """Current allocator state, or None when no CUDA device is available."""
    cuda = _cuda()
    if cuda is None:
        return None
    return GPUMemorySnapshot(
        allocated_bytes=int(cuda.memory_allocated()),
        reserved_bytes=int(cuda.memory_reserved()),
        peak_allocated_bytes=int(cuda.max_memory_allocated()),
    )


def empty_cache_after_failure() -> dict[str, JsonValue] | None:
    """Let the allocator reuse blocks held by a failed generation's tensors.

    The caller must `del` its own tensor references first; this only tells
    the allocator those blocks are free. Only called on the failure path:
    emptying the CUDA cache after every normal generation would be pure
    overhead with nothing to reclaim.
    """
    cuda = _cuda()
    if cuda is None:
        return None
    cuda.empty_cache()
    after = snapshot()
    return after.model_dump(mode="json") if after else None
