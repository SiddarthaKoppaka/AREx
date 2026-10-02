"""LM-requested retrospective checks of predictions against observed history."""

from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.retrodiction import RetrodictionRequest
from arc_agi_3.evidence import retrodict

from .io import EpisodeIO


def retrodiction_tool(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    """Report compatible/contradicting transitions; truth is never inferred."""
    query = RetrodictionRequest.model_validate(request.arguments)
    result = retrodict(io.events.read(), query)
    cited = (*result.compatible, *result.contradicting, *result.not_evaluable)
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"retrodiction": result.model_dump(mode="json")},
        evidence_refs=tuple(item.transition_event_id for item in cited),
    )
