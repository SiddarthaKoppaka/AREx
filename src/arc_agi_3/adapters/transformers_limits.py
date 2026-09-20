"""Explicit inference input limits for complete model prompts."""

from typing import Protocol


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
