"""Dependency-free, recorded hardware discovery."""

import os
import platform

from pydantic import Field

from arc_agi_3.contracts.base import Contract


class HardwareProfile(Contract):
    system: str
    machine: str
    processor: str
    cpu_count: int = Field(ge=1)
    memory_bytes: int | None = Field(default=None, ge=0)
    cuda_visible_devices: str | None = None
    kaggle: bool


def _memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return int(pages * page_size)


def detect_hardware() -> HardwareProfile:
    return HardwareProfile(
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 1,
        memory_bytes=_memory_bytes(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        kaggle="KAGGLE_KERNEL_RUN_TYPE" in os.environ,
    )
