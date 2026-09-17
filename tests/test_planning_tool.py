"""An LM-authored world model and search request flow through the trace."""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType, SearchMethod
from arc_agi_3.contracts.observation import Action
from arc_agi_3.planning import SearchBudget, SearchRequest
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel
from arc_agi_3.world_model import DeclarativeRule, SymbolicState, WorldModel


def test_search_method_objective_and_budget_originate_in_decision(
    tmp_path: Path,
) -> None:
    model = WorldModel(
        model_id="one-step",
        version=1,
        rules=(
            DeclarativeRule(
                rule_id="advance",
                action=Action(action_id=1),
                preconditions={"position": 0},
                effects={"position": 1},
            ),
        ),
    )
    search = SearchRequest(
        search_id="lm-search",
        model_id="one-step",
        model_version=1,
        method=SearchMethod.ASTAR,
        objective="Test the LM-selected position target.",
        start_state=SymbolicState.build({"position": 0}),
        goal_facts={"position": 1},
        heuristic_weights={"position": 1.0},
        budget=SearchBudget(max_node_expansions=3, max_depth=2),
    )
    plan = CognitiveDecision(
        assessment="Use my explicit model and target.",
        intent="Run the requested bounded A* search.",
        mode=DecisionMode.PLAN,
        world_model_updates=(model,),
        tool_requests=(
            ToolRequest(
                request_id="tool-search",
                tool_name="search_world_model",
                arguments=search.model_dump(mode="json"),
            ),
        ),
    )
    stop = CognitiveDecision(
        assessment="inspect", intent="stop", mode=DecisionMode.STOP
    )
    config = RunConfig(
        run_id="planning-tool", experiment_id="test", output_dir=tmp_path
    )
    runner = build_runner(config, FakeLineEnvironment(), ScriptedModel([plan, stop]))
    runner.run()
    events = runner.events.read()
    result = next(
        event for event in events if event.event_type is EventType.TOOL_RESULT
    )
    request = next(
        event for event in events if event.event_type is EventType.TOOL_REQUEST
    )
    request_record = request.payload["request"]
    result_record = result.payload["result"]
    assert isinstance(request_record, dict) and isinstance(result_record, dict)
    arguments = request_record["arguments"]
    output = result_record["output"]
    assert isinstance(arguments, dict) and isinstance(output, dict)
    search_output = output["search"]
    assert isinstance(search_output, dict)
    assert arguments["method"] == "astar"
    assert search_output["status"] == "found"
    assert EventType.WORLD_MODEL in {event.event_type for event in events}
    assert audit_agency_boundary(events) == ()
