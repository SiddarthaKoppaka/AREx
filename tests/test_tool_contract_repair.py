"""A malformed tool call is repaired as cheaply as malformed JSON, not later."""

from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from tests.generation_fixtures import ScriptedBackend, context


def decision_json(tool_name: str, arguments: dict[str, object]) -> str:
    decision = CognitiveDecision(
        assessment="investigate",
        intent="inspect the frame before acting",
        mode=DecisionMode.INVESTIGATE,
        tool_requests=(
            ToolRequest(request_id="r1", tool_name=tool_name, arguments=arguments),
        ),
    )
    return decision.model_dump_json()


BAD_CALL = decision_json(
    "inspect_frame_region",
    {"event_id": "e1", "region": {"layer": 0, "x": 0, "y": 0, "width": 2, "height": 2}},
)
GOOD_CALL = decision_json(
    "inspect_frame_region",
    {
        "event_id": "e1",
        "purpose": "zoom in",
        "token_budget": 1024,
        "min_row": 0,
        "max_row": 1,
        "min_col": 0,
        "max_col": 1,
    },
)


def test_malformed_tool_call_is_repaired_without_a_new_cognitive_turn() -> None:
    backend = ScriptedBackend([BAD_CALL, GOOD_CALL])
    response = StructuredModelAdapter(backend).decide(context())
    assert [a.role for a in response.attempts] == ["primary", "repair"]
    assert response.attempts[0].error_category == "tool_contract_violation"
    assert response.attempts[1].valid is True
    assert "CURRENT_OBSERVATION" not in backend.prompts[1]
    assert response.decision.tool_requests[0].arguments["min_row"] == 0


def test_repair_prompt_names_the_offending_tool_and_the_real_schema() -> None:
    backend = ScriptedBackend([BAD_CALL, GOOD_CALL])
    StructuredModelAdapter(backend).decide(context())
    repair_prompt = backend.prompts[1]
    assert "inspect_frame_region" in repair_prompt
    assert "min_row" in repair_prompt or "Field required" in repair_prompt


def test_unknown_tool_name_is_also_a_tool_contract_violation() -> None:
    unknown = decision_json("inspect_frame", {})
    backend = ScriptedBackend([unknown, GOOD_CALL])
    response = StructuredModelAdapter(backend).decide(context())
    assert response.attempts[0].error_category == "tool_contract_violation"
    assert "inspect_frame" in (response.attempts[0].validation_error or "")
