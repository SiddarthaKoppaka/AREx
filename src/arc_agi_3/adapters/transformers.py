"""Offline attached-weight inference through Hugging Face Transformers."""

from time import perf_counter
from typing import Any

from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.trace.canonical import canonical_json

from . import transformers_limits as limits
from .inference import BackendGeneration
from .model_invocation import run_model
from .transformers_config import TransformersConfig as TransformersConfig
from .transformers_loader import load_transformers


class TransformersBackend:
    def __init__(
        self,
        config: TransformersConfig,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        reporter: limits.StageReporter | None = None,
    ) -> None:
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be supplied together")
        self.config = config
        self.reporter = reporter
        if model is None:
            started = perf_counter()
            if reporter:
                reporter.stage("model_load_start", model=config.model_name)
            self.model, self.tokenizer = load_transformers(config)
            if reporter:
                reporter.stage(
                    "model_load_finish", seconds=round(perf_counter() - started, 3)
                )
        else:
            assert tokenizer is not None
            self.model, self.tokenizer = model, tokenizer

    @property
    def metadata(self) -> dict[str, Any]:
        return self.config.metadata()

    @property
    def effective_max_input_tokens(self) -> int:
        return limits.effective_input_limit(self.config, self.model)

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendGeneration:
        inputs = self._tokenize(prompt, json_schema)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > self.effective_max_input_tokens:
            raise limits.InputTokenLimitError(
                input_tokens, self.effective_max_input_tokens, self.config.model_name
            )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        return self._generate(inputs, input_tokens)

    def input_token_count(self, prompt: str, json_schema: dict[str, Any]) -> int:
        inputs = self._tokenize(prompt, json_schema)
        return int(inputs["input_ids"].shape[-1])

    def _tokenize(self, prompt: str, json_schema: dict[str, Any]) -> Any:
        grounded = prompt + "\nJSON_SCHEMA:\n" + canonical_json(json_schema)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": grounded}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

    def _generate(self, inputs: Any, input_tokens: int) -> BackendGeneration:
        output, latency_ms = run_model(
            self.model,
            inputs,
            input_tokens,
            max_new_tokens=self.config.max_new_tokens,
            max_time_seconds=self.config.max_time_seconds,
            reporter=self.reporter,
            memory_instrumentation=self.config.gpu_memory_instrumentation,
        )
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        output_tokens = len(output)
        del inputs, output
        return BackendGeneration(
            text=text,
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            ),
        )
