"""Model-facing CognitiveDecision JSON schema, stripped for repeated sending."""

from functools import cache
from typing import Any

from arc_agi_3.contracts.decision import CognitiveDecision

from .schema_render import strip_schema


@cache
def decision_schema() -> dict[str, Any]:
    stripped: dict[str, Any] = strip_schema(CognitiveDecision.model_json_schema())
    return stripped
