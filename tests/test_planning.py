"""Search executes the exact LM-authored method, target, and limits."""

from arc_agi_3.contracts.enums import SearchMethod
from arc_agi_3.contracts.observation import Action
from arc_agi_3.planning import SearchBudget, SearchRequest, run_search
from arc_agi_3.world_model import (
    DeclarativeRule,
    SymbolicState,
    WorldModel,
    WorldModelRuntime,
)


def line_model() -> WorldModel:
    return WorldModel(
        model_id="line",
        version=1,
        rules=(
            DeclarativeRule(
                rule_id="0",
                action=Action(action_id=1),
                preconditions={"position": 0},
                effects={"position": 1},
            ),
            DeclarativeRule(
                rule_id="1",
                action=Action(action_id=1),
                preconditions={"position": 1},
                effects={"position": 2},
            ),
        ),
    )


def request(method: SearchMethod, expansions: int = 10) -> SearchRequest:
    return SearchRequest(
        search_id=f"search-{method}",
        model_id="line",
        model_version=1,
        method=method,
        objective="Reach the LM-selected position 2 target.",
        start_state=SymbolicState.build({"position": 0}),
        goal_facts={"position": 2},
        heuristic_weights={"position": 1.0},
        budget=SearchBudget(max_node_expansions=expansions, max_depth=3),
    )


def test_bfs_and_astar_find_the_authored_target_deterministically() -> None:
    runtime = WorldModelRuntime(line_model())
    for method in (SearchMethod.BFS, SearchMethod.ASTAR):
        first = run_search(runtime, request(method))
        second = run_search(runtime, request(method))
        assert first == second
        assert first.status == "found"
        assert [action.action_id for action in first.actions] == [1, 1]
        assert first.states[-1].facts == {"position": 2}


def test_search_returns_partial_at_the_requested_limit() -> None:
    result = run_search(WorldModelRuntime(line_model()), request(SearchMethod.BFS, 1))
    assert result.status == "partial"
    assert result.node_expansions == 1
    assert result.actions == ()
