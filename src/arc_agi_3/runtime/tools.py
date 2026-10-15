"""Bounded deterministic tools invoked explicitly by an LM decision.

A tool call whose result depends only on its own arguments
(`evidence/tool_cache.py`) reuses an identical prior result instead of
recomputing it - but the request is still fully recorded, and the result
payload names the cache hit explicitly. Nothing is concealed; this only
avoids redundant computation for calls that must be exactly reproducible.
"""

from pydantic import JsonValue

from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.evidence.tool_cache import CACHEABLE_TOOLS
from arc_agi_3.evidence.tool_cues import tool_fingerprint

from .io import EpisodeIO
from .tool_handlers import execute_tool


def run_tool(
    io: EpisodeIO,
    request: ToolRequest,
    decision: EventEnvelope,
    step: int,
    observation: Observation,
) -> None:
    cacheable = request.tool_name in CACHEABLE_TOOLS
    fingerprint = tool_fingerprint(request)
    requested = io.append(
        EventType.TOOL_REQUEST,
        "tools",
        step,
        {"request": request.model_dump(mode="json")},
        (decision.event_id,),
    )
    cached = io.tool_cache.lookup(fingerprint) if cacheable else None
    payload: dict[str, JsonValue]
    if cached is not None:
        payload = {
            "result": cached.result.model_dump(mode="json"),
            "cache_hit": True,
            "cached_from_event_id": cached.source_event_id,
        }
    else:
        try:
            result = execute_tool(io, request, requested, step, observation)
        except (FileNotFoundError, KeyError, ValueError) as error:
            result = ToolResult(
                request_id=request.request_id,
                status="failed",
                output={"error_type": type(error).__name__, "message": str(error)},
            )
        if cacheable:
            io.tool_cache.store(fingerprint, result, requested.event_id)
        payload = {"result": result.model_dump(mode="json")}
    io.append(EventType.TOOL_RESULT, "tools", step, payload, (requested.event_id,))
