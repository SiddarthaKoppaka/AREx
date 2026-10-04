"""Bounded event-view tool handlers."""

from arc_agi_3.ablations import require_capability
from arc_agi_3.context import EventRetriever
from arc_agi_3.context.evidence_retrieval import retrieve_evidence
from arc_agi_3.context.frame_region import region_view
from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.evidence import EvidenceQuery, FrameRegionRequest
from arc_agi_3.contracts.retrieval import RetrievalQuery

from .io import EpisodeIO


def evidence_tool(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    require_capability(io.config.ablations.selective_retrieval, "retrieval")
    query = EvidenceQuery.model_validate(request.arguments)
    status, output, refs = retrieve_evidence(io.events.read(), query)
    return ToolResult(
        request_id=request.request_id,
        status=status,
        output={"evidence": output},
        evidence_refs=refs,
    )


def inspect_frame_region_tool(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    require_capability(io.config.ablations.selective_retrieval, "retrieval")
    query = FrameRegionRequest.model_validate(request.arguments)
    by_id = {event.event_id: event for event in io.events.read()}
    event = by_id.get(query.event_id)
    if event is None:
        raise ValueError(f"unknown evidence event ID: {query.event_id!r}")
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"region": region_view(event, query)},
        evidence_refs=(query.event_id,),
    )


def retrieve_events_tool(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    require_capability(io.config.ablations.selective_retrieval, "retrieval")
    query = RetrievalQuery.model_validate(request.arguments)
    retrieved = EventRetriever(io.events.read()).retrieve(query)
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"retrieval": retrieved.model_dump(mode="json")},
        evidence_refs=tuple(item.event_id for item in retrieved.matches),
    )
