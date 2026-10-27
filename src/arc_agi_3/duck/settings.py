"""Duck run settings, overridable from DUCK_* environment variables."""

import os
from typing import Literal

from arc_agi_3.contracts.base import Contract


class DuckSettings(Contract):
    model_path: str = ""
    backend: Literal["hf", "sglang", "vllm", "openrouter"] = "hf"
    model_name: str = "google/gemma-4-26b-a4b-it:free"
    thinking: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_tokens: int = 2048
    trace_path: str | None = None
    use_images: bool = False
    max_actions: int = 2000
    max_inner: int = 4
    port: int = 8000
    tp: int = 1
    context: int = 32768
    params_billion: float | None = None  # enables the VRAM preflight
    bits: int = 8

    @classmethod
    def from_env(cls) -> "DuckSettings":
        if "DUCK_MODEL_PATH" not in os.environ and (
            os.environ.get("DUCK_BACKEND") != "openrouter"
        ):
            raise RuntimeError("DUCK_MODEL_PATH is required")
        values = {
            name: os.environ[f"DUCK_{name.upper()}"]
            for name in cls.model_fields
            if f"DUCK_{name.upper()}" in os.environ
        }
        return cls.model_validate(values, strict=False)
