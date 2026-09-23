"""Bounded event-view tool handlers."""

from arc_agi_3.ablations import require_capability
from arc_agi_3.context.evidence_retrieval import retrieve_evidence
from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.evidence import EvidenceQuery

from .io import EpisodeIO


def evidence_tool(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    require_capability(io.config.ablations.selective_retrieval, "retrieval")
    arguments = dict(request.arguments)
    if request.tool_name == "inspect_frame_region":
        arguments["view"] = "region"
    query = EvidenceQuery.model_validate(arguments)
    status, output, refs = retrieve_evidence(io.events.read(), query)
    return ToolResult(
        request_id=request.request_id,
        status=status,
        output={"evidence": output},
        evidence_refs=refs,
    )
