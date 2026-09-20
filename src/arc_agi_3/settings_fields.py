"""Validated scalar fields for runtime profiles."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from arc_agi_3.contracts.base import Contract


class RuntimeFields(Contract):
    profile: str = "local_smoke"
    model_provider: Literal["transformers"] = "transformers"
    model_path: Path = Path("Qwen/Qwen3-8B")
    model_name: str = "Qwen/Qwen3-8B"
    model_digest: str = "unresolved"
    revision: str = "main"
    dtype: str = "auto"
    device_map: str = "auto"
    attention_implementation: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False
    quantization: Literal["none", "nf4"] = "none"
    max_input_tokens: int = Field(default=32768, gt=0)
    max_new_tokens: int = Field(default=1536, gt=0)
    generation_timeout_seconds: float = Field(default=180, gt=0)
    max_repairs: int = Field(default=1, ge=0)
    max_turns: int = Field(default=8, gt=0)
    max_actions: int = Field(default=40, gt=0)
    max_model_calls: int = Field(default=8, gt=0)
    input_token_budget: int = Field(default=400000, gt=0)
    output_token_budget: int = Field(default=20000, gt=0)
    wall_time_budget_seconds: float | None = Field(default=None, gt=0)
    checkpoint_interval: int = Field(default=1, gt=0)
    recent_event_limit: int = Field(default=12, gt=0)
    compact_context: bool = True
    output_dir: Path = Path("runs")
    log_dir: Path = Path("runs/logs")
    live_trace_mode: Literal["silent", "readable", "json"] = "readable"
    persist_invalid_model_output: bool = False
    model_staging: Literal["none", "copy_if_space"] = "none"
    model_cache_dir: Path | None = None
    require_gpu: bool = False
