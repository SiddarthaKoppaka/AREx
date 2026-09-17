"""Bounded deterministic tools invoked explicitly by an LM decision."""

from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope

from .io import EpisodeIO
from .tool_handlers import execute_tool


def run_tool(
    io: EpisodeIO, request: ToolRequest, decision: EventEnvelope, step: int
) -> None:
    requested = io.append(
        EventType.TOOL_REQUEST,
        "tools",
        step,
        {"request": request.model_dump(mode="json")},
        (decision.event_id,),
    )
    try:
        result = execute_tool(io, request, requested, step)
    except (KeyError, ValueError) as error:
        result = ToolResult(
            request_id=request.request_id,
            status="failed",
            output={"error_type": type(error).__name__, "message": str(error)},
        )
    io.append(
        EventType.TOOL_RESULT,
        "tools",
        step,
        {"result": result.model_dump(mode="json")},
        (requested.event_id,),
    )
