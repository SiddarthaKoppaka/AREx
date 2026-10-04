"""The model-facing tool schema and the execution-time contract are one thing."""

import pytest
from pydantic import ValidationError

from arc_agi_3.adapters.tool_contract_check import check_tool_requests
from arc_agi_3.adapters.tool_schemas import tool_schemas
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.evidence import FrameRegionRequest
from arc_agi_3.tool_registry import TOOL_ARGUMENTS


def test_every_registered_tool_has_a_rendered_schema() -> None:
    schemas = tool_schemas()
    assert set(schemas) == set(TOOL_ARGUMENTS)
    assert "schema_version" not in schemas["retrieve_evidence"]["properties"]


def test_inspect_frame_region_schema_matches_its_dedicated_contract() -> None:
    schema = tool_schemas()["inspect_frame_region"]
    fields = set(schema["properties"])
    assert fields == {
        "event_id",
        "purpose",
        "token_budget",
        "layer",
        "min_row",
        "max_row",
        "min_col",
        "max_col",
    }
    assert "region" not in fields and "event_ids" not in fields


def test_frame_region_request_rejects_the_wrong_shape_qwen_actually_tried() -> None:
    with pytest.raises(ValidationError):
        FrameRegionRequest.model_validate(
            {"event_id": "e1", "purpose": "look", "token_budget": 500, "region": {}}
        )
    with pytest.raises(ValidationError, match="min_row"):
        FrameRegionRequest.model_validate(
            {"event_ids": ["e1"], "purpose": "look", "token_budget": 500}
        )


def decision_with_tool(
    tool_name: str, arguments: dict[str, object]
) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="investigate",
        intent="inspect the frame",
        mode=DecisionMode.INVESTIGATE,
        tool_requests=(
            ToolRequest(request_id="r1", tool_name=tool_name, arguments=arguments),
        ),
    )


def test_check_tool_requests_accepts_a_correctly_shaped_call() -> None:
    decision = decision_with_tool(
        "inspect_frame_region",
        {
            "event_id": "e1",
            "purpose": "look",
            "token_budget": 500,
            "min_row": 0,
            "max_row": 3,
            "min_col": 0,
            "max_col": 3,
        },
    )
    check_tool_requests(decision)  # does not raise


def test_check_tool_requests_rejects_the_wrong_field_names_before_acceptance() -> None:
    from arc_agi_3.adapters.generation_errors import ToolContractError

    decision = decision_with_tool(
        "inspect_frame_region",
        {"event_id": "e1", "min_row": 0, "max_row": 1, "min_col": 0, "max_col": 1},
    )
    with pytest.raises(ToolContractError, match="inspect_frame_region"):
        check_tool_requests(decision)


def test_check_tool_requests_rejects_an_unregistered_tool_name() -> None:
    from arc_agi_3.adapters.generation_errors import ToolContractError

    decision = decision_with_tool("inspect_frame", {})
    with pytest.raises(ToolContractError, match="unknown tool"):
        check_tool_requests(decision)
