"""Persist LM-authored compact artifacts with local evidence provenance."""

from typing import Literal

from pydantic import Field, JsonValue

from arc_agi_3.ablations import require_capability
from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.trace.canonical import canonical_json

from .io import EpisodeIO


class ArchiveArtifactRequest(Contract):
    purpose: str = Field(min_length=1, max_length=512)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=20)
    content: dict[str, JsonValue]


class ReadArtifactRequest(Contract):
    artifact_ref: str
    purpose: str = Field(min_length=1, max_length=240)
    token_budget: int = Field(ge=128, le=65536)
    keys: tuple[str, ...] = ()


def archive_artifact(
    io: EpisodeIO, request: ToolRequest, requested: EventEnvelope
) -> ToolResult:
    require_capability(io.config.ablations.persistent_memory, "persistent_memory")
    arguments = ArchiveArtifactRequest.model_validate(request.arguments)
    known = {
        event.event_id
        for event in io.events.read()
        if event.sequence < requested.sequence
    }
    if not set(arguments.evidence_refs) <= known:
        raise ValueError("artifact evidence must reference prior events")
    reference = io.events.artifacts.archive_document(arguments.model_dump(mode="json"))
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"artifact_ref": reference, "purpose": arguments.purpose},
        evidence_refs=arguments.evidence_refs,
    )


def retrieve_artifact(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    require_capability(io.config.ablations.persistent_memory, "persistent_memory")
    query = ReadArtifactRequest.model_validate(request.arguments)
    artifact = io.events.artifacts.read(query.artifact_ref)
    available = sorted(artifact)
    keys = query.keys or tuple(available)
    if not set(keys) <= set(available):
        raise ValueError("requested artifact keys are unavailable")
    selected = {key: artifact[key] for key in keys}
    visible_keys: list[JsonValue] = list(available)
    output: dict[str, JsonValue] = {
        "artifact_ref": query.artifact_ref,
        "purpose": query.purpose,
        "available_keys": visible_keys,
        "content": selected,
    }
    status: Literal["complete", "partial"] = "complete"
    if len(canonical_json(output).encode("utf-8")) > query.token_budget:
        output["content"] = {}
        output["omitted_keys"] = list(keys)
        status = "partial"
    if len(canonical_json(output).encode("utf-8")) > query.token_budget:
        raise ValueError("artifact metadata exceeds requested token budget")
    return ToolResult(request_id=request.request_id, status=status, output=output)
