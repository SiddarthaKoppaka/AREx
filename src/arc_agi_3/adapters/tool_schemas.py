"""Stripped, model-facing JSON schema for every registered tool.

Rendered from the same registry the generation loop validates
`tool_requests` against, so the model is shown exactly the contract it will
be checked against, not prose that can silently drift from it.
"""

from functools import cache
from typing import Any

from arc_agi_3.tool_registry import TOOL_ARGUMENTS

from .schema_render import strip_schema


@cache
def tool_schemas() -> dict[str, dict[str, Any]]:
    return {
        name: strip_schema(model.model_json_schema())
        for name, model in TOOL_ARGUMENTS.items()
    }
