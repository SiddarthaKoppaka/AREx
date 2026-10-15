"""Every registered tool is explained in prose or in its own schema fields,
and the model-facing prompt/schema sizes stay within a bounded budget."""

import json

from arc_agi_3.adapters.prompt_text import TOOLS
from arc_agi_3.adapters.tool_schemas import tool_schemas
from arc_agi_3.tool_registry import TOOL_ARGUMENTS


def _has_field_descriptions(name: str) -> bool:
    properties = tool_schemas()[name].get("properties", {})
    return bool(properties) and all(
        "description" in field for field in properties.values()
    )


def test_every_registered_tool_is_named_in_prose_or_fully_described() -> None:
    for name in TOOL_ARGUMENTS:
        assert name in TOOLS or _has_field_descriptions(name), name


def test_drig_and_world_model_identifiers_carry_a_description() -> None:
    exploration = tool_schemas()["evaluate_exploration"]
    probe = exploration["$defs"]["ProbeEstimate"]["properties"]
    assert probe["drig"]["description"]
    simulate = tool_schemas()["simulate_world_model"]["properties"]
    assert simulate["model_id"]["description"]
    search = tool_schemas()["search_world_model"]["properties"]
    assert search["model_id"]["description"]
    assert search["goal_facts"]["description"]


def test_prompt_and_schema_sizes_stay_within_a_bounded_budget() -> None:
    assert len(TOOLS) < 3500
    assert len(json.dumps(tool_schemas()).encode()) < 20_000
