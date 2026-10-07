"""Implement deterministic tool capabilities selected by the LM."""

from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation

from .artifact_tools import archive_artifact, retrieve_artifact
from .epistemic_tools import retrodiction_tool
from .evidence_tools import (
    evidence_tool,
    inspect_frame_region_tool,
    retrieve_events_tool,
)
from .io import EpisodeIO
from .specialist_compare_tool import compare_specialist_reports_tool
from .specialist_tools import consult_specialist_tool
from .world_model_tools import (
    evaluate_exploration_tool,
    record_exploration_outcome_tool,
    search_world_model_tool,
    simulate_world_model_tool,
)


def execute_tool(
    io: EpisodeIO,
    request: ToolRequest,
    requested: EventEnvelope,
    step: int,
    observation: Observation,
) -> ToolResult:
    if request.tool_name == "consult_specialist":
        return consult_specialist_tool(io, request, requested, observation, step)
    if request.tool_name == "compare_specialist_reports":
        return compare_specialist_reports_tool(io, request, requested, step)
    if request.tool_name == "archive_artifact":
        return archive_artifact(io, request, requested)
    if request.tool_name == "retrieve_artifact":
        return retrieve_artifact(io, request)
    if request.tool_name == "retrieve_evidence":
        return evidence_tool(io, request)
    if request.tool_name == "inspect_frame_region":
        return inspect_frame_region_tool(io, request)
    if request.tool_name == "check_prediction_history":
        return retrodiction_tool(io, request)
    if request.tool_name == "retrieve_events":
        return retrieve_events_tool(io, request)
    if request.tool_name == "simulate_world_model":
        return simulate_world_model_tool(io, request, requested, step)
    if request.tool_name == "search_world_model":
        return search_world_model_tool(io, request, requested, step)
    if request.tool_name == "evaluate_exploration":
        return evaluate_exploration_tool(io, request)
    if request.tool_name == "record_exploration_outcome":
        return record_exploration_outcome_tool(io, request)
    raise ValueError(f"unsupported tool {request.tool_name!r}")
