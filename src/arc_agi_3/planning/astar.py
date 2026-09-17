"""Bounded deterministic A* with an LM-authored heuristic."""

import heapq
from itertools import count

from arc_agi_3.contracts.observation import Action
from arc_agi_3.world_model.contracts import SymbolicState
from arc_agi_3.world_model.runtime import WorldModelRuntime

from .common import heuristic, is_goal
from .contracts import SearchRequest, SearchResult, SearchStatus

Node = tuple[
    float, int, float, SymbolicState, tuple[Action, ...], tuple[SymbolicState, ...]
]


def a_star(runtime: WorldModelRuntime, request: SearchRequest) -> SearchResult:
    serial = count()
    start: Node = (
        heuristic(request.start_state, request),
        next(serial),
        0.0,
        request.start_state,
        (),
        (request.start_state,),
    )
    heap = [start]
    best = {request.start_state.state_hash: 0.0}
    expansions = reached = 0
    depth_pruned = False
    while heap and expansions < request.budget.max_node_expansions:
        _, _, cost, state, actions, states = heapq.heappop(heap)
        depth = len(actions)
        reached = max(reached, depth)
        if cost > best.get(state.state_hash, float("inf")):
            continue
        if is_goal(state, request):
            return _result(
                request, "found", actions, states, cost, expansions, reached, heap
            )
        if depth >= request.budget.max_depth:
            depth_pruned = True
            continue
        expansions += 1
        for rule, after in runtime.transitions(state):
            new_cost = cost + rule.cost
            if new_cost >= best.get(after.state_hash, float("inf")):
                continue
            best[after.state_hash] = new_cost
            priority = new_cost + heuristic(after, request)
            heapq.heappush(
                heap,
                (
                    priority,
                    next(serial),
                    new_cost,
                    after,
                    (*actions, rule.action),
                    (*states, after),
                ),
            )
    status: SearchStatus = "partial" if heap or depth_pruned else "exhausted"
    return _result(request, status, (), (), 0.0, expansions, reached, heap)


def _result(
    request: SearchRequest,
    status: SearchStatus,
    actions: tuple[Action, ...],
    states: tuple[SymbolicState, ...],
    cost: float,
    expansions: int,
    depth: int,
    heap: list[Node],
) -> SearchResult:
    return SearchResult(
        search_id=request.search_id,
        method=request.method,
        status=status,
        actions=actions,
        states=states,
        total_cost=cost,
        node_expansions=expansions,
        reached_depth=depth,
        frontier_state_hashes=tuple(node[3].state_hash for node in sorted(heap)),
    )
