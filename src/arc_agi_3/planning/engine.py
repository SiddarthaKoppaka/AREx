"""Dispatch exactly the search method requested by the LM."""

from arc_agi_3.contracts.enums import SearchMethod
from arc_agi_3.world_model.runtime import WorldModelRuntime

from .astar import a_star
from .bfs import breadth_first
from .contracts import SearchRequest, SearchResult


def run_search(runtime: WorldModelRuntime, request: SearchRequest) -> SearchResult:
    if (request.model_id, request.model_version) != (
        runtime.model.model_id,
        runtime.model.version,
    ):
        raise ValueError("search request targets a different model version")
    if request.method is SearchMethod.BFS:
        return breadth_first(runtime, request)
    if request.method is SearchMethod.ASTAR:
        return a_star(runtime, request)
    raise ValueError(f"unsupported search method {request.method}")
