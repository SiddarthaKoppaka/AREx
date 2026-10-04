"""Argument contracts for the LM-facing artifact tools."""

from pydantic import Field, JsonValue

from .base import Contract


class ArchiveArtifactRequest(Contract):
    purpose: str = Field(min_length=1, max_length=512)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=20)
    content: dict[str, JsonValue]


class ReadArtifactRequest(Contract):
    artifact_ref: str
    purpose: str = Field(min_length=1, max_length=240)
    token_budget: int = Field(ge=128, le=65536)
    keys: tuple[str, ...] = ()
