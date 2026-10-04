"""Validated, bounded structured-output adapter.

`max_repairs` is hard-capped at `GENERATION_CEILING` so a bad parse can never
balloon into dozens of full-cost regenerations; see `generation_loop` for how
a repair attempt differs from the primary cognitive generation.
"""

from pydantic import JsonValue

from arc_agi_3.contracts.decision import AgentContext, ModelResponse

from .generation_loop import run_generation
from .inference import InferenceBackend
from .structured_output import StructuredOutputError as StructuredOutputError
from .structured_output import parse_json_object as parse_json_object

GENERATION_CEILING = 2
PREVIEW_CHARS = 300


class StructuredModelAdapter:
    def __init__(
        self,
        backend: InferenceBackend,
        max_repairs: int = GENERATION_CEILING,
        persist_invalid_output: bool = False,
        preview_chars: int = PREVIEW_CHARS,
    ) -> None:
        if not 0 <= max_repairs <= GENERATION_CEILING:
            raise ValueError(f"max_repairs must be between 0 and {GENERATION_CEILING}")
        self.backend, self.max_repairs = backend, max_repairs
        self.persist_invalid_output = persist_invalid_output
        self.preview_chars = preview_chars

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {
            "adapter": "structured-model",
            "max_repairs": self.max_repairs,
            "generation_ceiling": GENERATION_CEILING,
            "persist_invalid_output": self.persist_invalid_output,
            "backend": self.backend.metadata,
        }

    def decide(self, context: AgentContext) -> ModelResponse:
        return run_generation(
            self.backend,
            context,
            self.max_repairs,
            persist_invalid_output=self.persist_invalid_output,
            preview_chars=self.preview_chars,
        )
