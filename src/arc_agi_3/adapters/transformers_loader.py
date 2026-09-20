"""Optional Transformers dependency loading and quantization options."""

from importlib import import_module
from typing import Any, Literal, Protocol

QuantizationMode = Literal["none", "nf4"]


class LoadConfig(Protocol):
    model_path: str
    dtype: str
    device_map: str
    attention_implementation: str | None
    local_files_only: bool
    trust_remote_code: bool
    revision: str
    quantization: QuantizationMode


def load_transformers(config: LoadConfig) -> tuple[Any, Any]:
    try:
        transformers = import_module("transformers")
    except ImportError as error:
        raise RuntimeError(
            "install a Qwen3-compatible Transformers build for inference"
        ) from error
    common = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "revision": config.revision,
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_path, **common)
    model_options: dict[str, Any] = {
        "dtype": config.dtype,
        "device_map": config.device_map,
    }
    if config.attention_implementation is not None:
        model_options["attn_implementation"] = config.attention_implementation
    if config.quantization == "nf4":
        torch = import_module("torch")
        model_options["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_path, **model_options, **common
    )
    return model.eval(), tokenizer
