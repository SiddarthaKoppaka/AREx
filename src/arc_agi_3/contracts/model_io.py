"""Model-call usage, attempt, and response records."""

from pydantic import Field

from .base import Contract
from .decision_core import CognitiveDecision


class ModelUsage(Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class ModelAttempt(Contract):
    attempt: int = Field(ge=1)
    output_hash: str
    valid: bool
    validation_error: str | None = None
    output_preview: str | None = None


class PromptReport(Contract):
    """Measured prompt size for the final attempt, before and after compaction."""

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
