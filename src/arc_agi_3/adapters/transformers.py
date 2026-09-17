"""Offline attached-weight inference through Hugging Face Transformers."""

from importlib import import_module
from time import perf_counter
from typing import Any

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.trace.canonical import canonical_json

from .inference import BackendGeneration


class TransformersConfig(Contract):
    model_path: str
    model_name: str = "Qwen/Qwen3.5-9B"
    model_digest: str = "unresolved"
    max_new_tokens: int = Field(default=2048, gt=0)
    dtype: str = "auto"
    device_map: str = "auto"


class TransformersBackend:
    def __init__(
        self,
        config: TransformersConfig,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        self.config = config
        if model is None:
            self.model, self.processor = self._load()
        else:
            assert processor is not None
            self.model, self.processor = model, processor

    def _load(self) -> tuple[Any, Any]:
        try:
            transformers = import_module("transformers")
        except ImportError as error:
            raise RuntimeError(
                "install a Qwen3.5-compatible Transformers build for Kaggle inference"
            ) from error
        common = {"local_files_only": True, "trust_remote_code": False}
        processor = transformers.AutoProcessor.from_pretrained(
            self.config.model_path, **common
        )
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.config.model_path,
            dtype=self.config.dtype,
            device_map=self.config.device_map,
            **common,
        )
        return model.eval(), processor

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "transformers",
            "model": self.config.model_name,
            "model_digest": self.config.model_digest,
            "max_new_tokens": self.config.max_new_tokens,
            "dtype": self.config.dtype,
            "device_map": self.config.device_map,
        }

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendGeneration:
        grounded = prompt + "\nJSON_SCHEMA:\n" + canonical_json(json_schema)
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": grounded}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        started = perf_counter()
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )[0][input_tokens:]
        latency_ms = round((perf_counter() - started) * 1000)
        return BackendGeneration(
            text=self.processor.decode(output, skip_special_tokens=True),
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=len(output),
                latency_ms=latency_ms,
            ),
        )
