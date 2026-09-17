"""Deterministic transition and verification results."""

from pydantic import Field, JsonValue

from .base import Contract


class StateDelta(Contract):
    before_hash: str
    after_hash: str
    changed_cells: int = Field(ge=0)
    metadata_changes: dict[str, tuple[JsonValue, JsonValue]] = Field(
        default_factory=dict
    )


class VerificationResult(Contract):
    checked: bool
    passed: bool
    mismatches: tuple[str, ...] = ()
    delta: StateDelta
