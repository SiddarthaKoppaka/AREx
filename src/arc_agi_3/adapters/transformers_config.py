"""Configuration for offline Transformers inference."""

from pydantic import Field

from arc_agi_3.contracts.base import Contract

from .transformers_loader import QuantizationMode


class TransformersConfig(Contract):
    model_path: str
    model_name: str = "Qwen/Qwen3-8B"
    model_digest: str = "unresolved"
    max_new_tokens: int = Field(default=2048, gt=0)
    max_input_tokens: int = Field(default=30720, gt=0)
    max_context_tokens: int = Field(default=32768, gt=0)
    soft_input_limit: int = Field(default=12000, gt=0)
    compaction_pressure_start: int = Field(default=12000, gt=0)
    active_context_target: int = Field(default=16000, gt=0)
    template_and_generation_margin: int = Field(default=512, ge=0)
    max_time_seconds: float = Field(default=180, gt=0)
    dtype: str = "auto"
    device_map: str = "auto"
    attention_implementation: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False
    revision: str = "main"
    quantization: QuantizationMode = "none"

    def metadata(self) -> dict[str, object]:
        keys = (
            "model_name",
            "model_digest",
            "max_new_tokens",
            "max_input_tokens",
            "max_context_tokens",
            "soft_input_limit",
            "compaction_pressure_start",
            "active_context_target",
            "template_and_generation_margin",
            "max_time_seconds",
            "dtype",
            "device_map",
            "attention_implementation",
            "local_files_only",
            "trust_remote_code",
            "revision",
            "quantization",
        )
        return {"provider": "transformers", "model": self.model_name} | {
            key: getattr(self, key) for key in keys if key != "model_name"
        }
