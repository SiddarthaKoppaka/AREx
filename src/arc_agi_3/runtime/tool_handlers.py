"""Implement deterministic tool capabilities selected by the LM."""

from arc_agi_3.ablations import require_capability
from arc_agi_3.context import EventRetriever
from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.enums import Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.retrieval import RetrievalQuery
from arc_agi_3.exploration import ExplorationRequest, evaluate_exploration
from arc_agi_3.exploration.contracts import ExplorationOutcome
from arc_agi_3.planning import SearchRequest, run_search
from arc_agi_3.world_model import SimulationRequest, WorldModelRuntime

from .artifact_tools import archive_artifact, retrieve_artifact
from .evidence_tools import evidence_tool
from .io import EpisodeIO


def execute_tool(
    io: EpisodeIO, request: ToolRequest, requested: EventEnvelope, step: int
) -> ToolResult:
    if request.tool_name == "archive_artifact":
        return archive_artifact(io, request, requested)
    if request.tool_name == "retrieve_artifact":
        return retrieve_artifact(io, request)
    if request.tool_name in {"retrieve_evidence", "inspect_frame_region"}:
        return evidence_tool(io, request)
    if request.tool_name == "retrieve_events":
        require_capability(io.config.ablations.selective_retrieval, "retrieval")
        query = RetrievalQuery.model_validate(request.arguments)
        retrieved = EventRetriever(io.events.read()).retrieve(query)
        return ToolResult(
            request_id=request.request_id,
            status="complete",
            output={"retrieval": retrieved.model_dump(mode="json")},
            evidence_refs=tuple(item.event_id for item in retrieved.matches),
        )
    if request.tool_name == "simulate_world_model":
        require_capability(io.config.ablations.world_models, "world_models")
        simulation = SimulationRequest.model_validate(request.arguments)
        model = io.workspace.world_models.get(
            simulation.model_id, simulation.model_version
        )
        simulation_result = WorldModelRuntime(model).simulate(simulation)
        io.spend(
            Resource.SIMULATIONS,
            len(simulation_result.steps),
            step,
            (requested.event_id,),
        )
        return ToolResult(
            request_id=request.request_id,
            status=simulation_result.status,
            output={"simulation": simulation_result.model_dump(mode="json")},
        )
    if request.tool_name == "search_world_model":
        require_capability(io.config.ablations.search, "search")
        require_capability(io.config.ablations.world_models, "world_models")
        search = SearchRequest.model_validate(request.arguments)
        model = io.workspace.world_models.get(search.model_id, search.model_version)
        search_result = run_search(WorldModelRuntime(model), search)
        if search_result.node_expansions:
            io.spend(
                Resource.SEARCH_NODES,
                search_result.node_expansions,
                step,
                (requested.event_id,),
            )
        return ToolResult(
            request_id=request.request_id,
            status="complete" if search_result.status == "found" else "partial",
            output={"search": search_result.model_dump(mode="json")},
        )
    if request.tool_name == "evaluate_exploration":
        evaluation = ExplorationRequest.model_validate(request.arguments)
        if not evaluation.recent_marginal_values:
            evaluation = evaluation.model_copy(
                update={
                    "recent_marginal_values": io.workspace.exploration.marginal_values
                }
            )
        exploration_result = evaluate_exploration(evaluation, io.workspace.exploration)
        return ToolResult(
            request_id=request.request_id,
            status="complete",
            output={"exploration": exploration_result.model_dump(mode="json")},
        )
    if request.tool_name == "record_exploration_outcome":
        outcome = ExplorationOutcome.model_validate(request.arguments)
        io.workspace.exploration.record(outcome.state_hash, outcome.marginal_value)
        return ToolResult(
            request_id=request.request_id,
            status="complete",
            output={"history": io.workspace.exploration.snapshot()},
        )
    raise ValueError(f"unsupported tool {request.tool_name!r}")
