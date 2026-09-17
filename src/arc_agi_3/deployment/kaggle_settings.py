"""Centralized settings passed from the generated Kaggle notebook."""

import os
from pathlib import Path

from pydantic import Field

from arc_agi_3.contracts.base import Contract


class KaggleSettings(Contract):
    model_path: Path
    output_dir: Path = Path("/kaggle/working/arex-runs")
    model_name: str = "Qwen/Qwen3-8B"
    model_digest: str = "unresolved"
    max_actions: int = Field(default=40, gt=0)
    max_model_calls: int = Field(default=4, gt=0)
    max_input_tokens: int = Field(default=200_000, gt=0)
    max_output_tokens: int = Field(default=8_000, gt=0)
    max_new_tokens: int = Field(default=2_048, gt=0)
    max_repairs: int = Field(default=1, ge=0)
    require_gpu: bool = True

    @classmethod
    def from_env(cls) -> "KaggleSettings":
        model_path = os.environ.get("AREX_MODEL_PATH")
        if not model_path:
            raise RuntimeError("AREX_MODEL_PATH is required")
        return cls(
            model_path=Path(model_path),
            output_dir=Path(os.getenv("AREX_OUTPUT_DIR", "/kaggle/working/arex-runs")),
            max_actions=int(os.getenv("AREX_MAX_ACTIONS", "40")),
            max_model_calls=int(os.getenv("AREX_MAX_MODEL_CALLS", "4")),
            max_new_tokens=int(os.getenv("AREX_MAX_NEW_TOKENS", "2048")),
            max_repairs=int(os.getenv("AREX_MAX_REPAIRS", "1")),
        )
