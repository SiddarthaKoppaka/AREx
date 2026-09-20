"""Configuration for offline Transformers inference."""

from pydantic import Field

from arc_agi_3.contracts.base import Contract

from .transformers_loader import QuantizationMode


class TransformersConfig(Contract):
    model_path: str
    model_name: str = "Qwen/Qwen3-8B"
    model_digest: str = "unresolved"
    max_new_tokens: int = Field(default=2048, gt=0)
    max_input_tokens: int = Field(default=32768, gt=0)
    max_time_seconds: float = Field(default=180, gt=0)
    dtype: str = "auto"
    device_map: str = "auto"
    attention_implementation: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False
    revision: str = "main"
    quantization: QuantizationMode = "none"
