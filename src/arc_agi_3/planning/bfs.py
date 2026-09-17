"""Bounded deterministic breadth-first search."""

from collections import deque

from arc_agi_3.contracts.observation import Action
from arc_agi_3.world_model.contracts import SymbolicState
from arc_agi_3.world_model.runtime import WorldModelRuntime

from .common import is_goal
from .contracts import SearchRequest, SearchResult, SearchStatus


def breadth_first(runtime: WorldModelRuntime, request: SearchRequest) -> SearchResult:
    Node = tuple[SymbolicState, tuple[Action, ...], tuple[SymbolicState, ...], float]
    queue: deque[Node] = deque([(request.start_state, (), (request.start_state,), 0.0)])
    visited = {request.start_state.state_hash}
    expansions = reached = 0
    depth_pruned = False
    while queue and expansions < request.budget.max_node_expansions:
        state, actions, states, cost = queue.popleft()
        depth = len(actions)
        reached = max(reached, depth)
        if is_goal(state, request):
            return _result(
                request, "found", actions, states, cost, expansions, reached, queue
            )
        if depth >= request.budget.max_depth:
            depth_pruned = True
            continue
        expansions += 1
        for rule, after in runtime.transitions(state):
            if after.state_hash in visited:
                continue
            visited.add(after.state_hash)
            queue.append(
                (after, (*actions, rule.action), (*states, after), cost + rule.cost)
            )
    status: SearchStatus = "partial" if queue or depth_pruned else "exhausted"
    return _result(request, status, (), (), 0.0, expansions, reached, queue)


def _result(
    request: SearchRequest,
    status: SearchStatus,
    actions: tuple[Action, ...],
    states: tuple[SymbolicState, ...],
    cost: float,
    expansions: int,
    depth: int,
    queue: deque[
        tuple[SymbolicState, tuple[Action, ...], tuple[SymbolicState, ...], float]
    ],
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
        frontier_state_hashes=tuple(node[0].state_hash for node in queue),
    )
