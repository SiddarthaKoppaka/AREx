"""Shared exact predicates and result construction for search."""

from arc_agi_3.world_model.contracts import SymbolicState

from .contracts import SearchRequest


def is_goal(state: SymbolicState, request: SearchRequest) -> bool:
    return all(
        state.facts.get(key) == value for key, value in request.goal_facts.items()
    )


def heuristic(state: SymbolicState, request: SearchRequest) -> float:
    return sum(
        request.heuristic_weights.get(key, 0.0)
        for key, value in request.goal_facts.items()
        if state.facts.get(key) != value
    )
