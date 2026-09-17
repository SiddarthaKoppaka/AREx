"""LM-authored prediction and bounded execution authorization."""

from pydantic import Field, model_validator

from .base import Contract
from .enums import EnvironmentState
from .observation import Action


class ExpectedOutcome(Contract):
    observation_hash: str | None = None
    state: EnvironmentState | None = None
    min_levels_completed: int | None = Field(default=None, ge=0)
    max_changed_cells: int | None = Field(default=None, ge=0)


class ChunkStep(Contract):
    action: Action
    required_before_hash: str | None = None
    expected_outcome: ExpectedOutcome | None = None


class ActionChunk(Contract):
    chunk_id: str
    steps: tuple[ChunkStep, ...] = Field(min_length=1)
    checkpoint_after_each: bool = False
    stop_on_mismatch: bool = True
    soft_interrupt_changed_cells: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bounded(self) -> "ActionChunk":
        if len(self.steps) > 64:
            raise ValueError("an action chunk may authorize at most 64 steps")
        return self


class DivergenceRecord(Contract):
    event_id: str
    sequence: int = Field(ge=0)
    step_id: int = Field(ge=0)
    reason: str
    evidence_refs: tuple[str, ...] = ()
