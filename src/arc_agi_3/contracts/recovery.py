"""LM-selected recovery operations and deterministic evidence projections."""

from string import ascii_letters, digits

from pydantic import Field, model_validator

from .base import Contract
from .enums import RecoveryOperation
from .retrieval import ContextEvent


class RecoveryRequest(Contract):
    operation: RecoveryOperation
    diagnosis: str
    divergence_event_id: str | None = None
    checkpoint_id: str | None = None
    new_branch_id: str | None = None

    @model_validator(mode="after")
    def required_targets(self) -> "RecoveryRequest":
        if self.operation is RecoveryOperation.FORK_CHECKPOINT and (
            not self.checkpoint_id or not self.new_branch_id
        ):
            raise ValueError("checkpoint fork requires checkpoint_id and new_branch_id")
        for value in (self.checkpoint_id, self.new_branch_id):
            if value and any(
                character not in ascii_letters + digits + "._-" for character in value
            ):
                raise ValueError("recovery identifiers contain unsafe characters")
        return self


class RecoveryEvidence(Contract):
    divergence_event_id: str
    divergence_sequence: int = Field(ge=0)
    reason: str
    local_events: tuple[ContextEvent, ...]
    checkpoint_ids: tuple[str, ...] = ()


class BranchRecord(Contract):
    branch_id: str
    parent_branch_id: str | None = None
    checkpoint_id: str | None = None
    status: str
