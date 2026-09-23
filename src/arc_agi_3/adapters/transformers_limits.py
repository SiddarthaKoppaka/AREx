"""Explicit inference input limits for complete model prompts."""

from typing import Protocol


class ContextLimitConfig(Protocol):
    max_input_tokens: int
    max_context_tokens: int
    max_new_tokens: int
    template_and_generation_margin: int


def effective_input_limit(config: ContextLimitConfig, model: object) -> int:
    configured = config.max_input_tokens
    declared = config.max_context_tokens
    model_config = getattr(model, "config", None)
    actual = getattr(model_config, "max_position_embeddings", declared)
    window = min(declared, actual) if isinstance(actual, int) else declared
    reserved = config.max_new_tokens + config.template_and_generation_margin
    return min(configured, window - reserved)


class StageReporter(Protocol):
    def stage(self, name: str, **details: object) -> None: ...


class InputTokenLimitError(ValueError):
    def __init__(
        self, actual: int, limit: int, model: str, context: str | None = None
    ) -> None:
        self.actual = actual
        self.limit = limit
        self.model = model
        self.context = context
        location = f" ({context})" if context else ""
        super().__init__(
            f"{model}{location} input has {actual} tokens; configured limit is {limit}"
        )
