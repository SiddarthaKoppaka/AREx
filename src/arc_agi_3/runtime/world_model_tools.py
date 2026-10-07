"""World-model and exploration tool handlers, split out of `tool_handlers.py`
to match how every other tool's handler already lives in its own module."""

from arc_agi_3.ablations import require_capability
from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.enums import Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.exploration import ExplorationRequest, evaluate_exploration
from arc_agi_3.exploration.contracts import ExplorationOutcome
from arc_agi_3.planning import SearchRequest, run_search
from arc_agi_3.world_model import SimulationRequest, WorldModelRuntime

from .io import EpisodeIO


def simulate_world_model_tool(
    io: EpisodeIO, request: ToolRequest, requested: EventEnvelope, step: int
) -> ToolResult:
    require_capability(io.config.ablations.world_models, "world_models")
    simulation = SimulationRequest.model_validate(request.arguments)
    model = io.workspace.world_models.get(simulation.model_id, simulation.model_version)
    result = WorldModelRuntime(model).simulate(simulation)
    io.spend(Resource.SIMULATIONS, len(result.steps), step, (requested.event_id,))
    return ToolResult(
        request_id=request.request_id,
        status=result.status,
        output={"simulation": result.model_dump(mode="json")},
    )


def search_world_model_tool(
    io: EpisodeIO, request: ToolRequest, requested: EventEnvelope, step: int
) -> ToolResult:
    require_capability(io.config.ablations.search, "search")
    require_capability(io.config.ablations.world_models, "world_models")
    search = SearchRequest.model_validate(request.arguments)
    model = io.workspace.world_models.get(search.model_id, search.model_version)
    result = run_search(WorldModelRuntime(model), search)
    if result.node_expansions:
        io.spend(
            Resource.SEARCH_NODES, result.node_expansions, step, (requested.event_id,)
        )
    return ToolResult(
        request_id=request.request_id,
        status="complete" if result.status == "found" else "partial",
        output={"search": result.model_dump(mode="json")},
    )


def evaluate_exploration_tool(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    evaluation = ExplorationRequest.model_validate(request.arguments)
    if not evaluation.recent_marginal_values:
        evaluation = evaluation.model_copy(
            update={"recent_marginal_values": io.workspace.exploration.marginal_values}
        )
    result = evaluate_exploration(evaluation, io.workspace.exploration)
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"exploration": result.model_dump(mode="json")},
    )


def record_exploration_outcome_tool(io: EpisodeIO, request: ToolRequest) -> ToolResult:
    outcome = ExplorationOutcome.model_validate(request.arguments)
    io.workspace.exploration.record(outcome.state_hash, outcome.marginal_value)
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"history": io.workspace.exploration.snapshot()},
    )
