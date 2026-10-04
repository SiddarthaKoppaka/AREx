"""Model-call usage, attempt, and response records."""

from pydantic import Field

from .base import Contract
from .decision_core import CognitiveDecision
from .generation import ErrorCategory, GenerationRole


class ModelUsage(Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class ModelAttempt(Contract):
    """One backend generation. `role` separates cognition from format repair."""

    attempt: int = Field(ge=1)
    role: GenerationRole = "primary"
    output_hash: str
    valid: bool
    char_count: int = Field(default=0, ge=0)
    generation_input_tokens: int = Field(default=0, ge=0)
    generation_output_tokens: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    json_candidate_detected: bool = True
    error_category: ErrorCategory | None = None
    validation_error: str | None = None
    output_preview: str | None = None
    output_preview_tail: str | None = None


class PromptReport(Contract):
    """Measured prompt size for the primary attempt, before and after compaction."""

    tokens_before_compaction: int = Field(ge=0)
    tokens_after_compaction: int = Field(ge=0)
    compacted: bool
    recent_events_after: int = Field(ge=0)
    episodic_items_after: int = Field(ge=0)


class ModelResponse(Contract):
    decision: CognitiveDecision
    usage: ModelUsage = Field(default_factory=ModelUsage)
    attempts: tuple[ModelAttempt, ...] = ()
    prompt: PromptReport | None = None
