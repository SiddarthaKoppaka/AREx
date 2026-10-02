"""Model-facing CognitiveDecision JSON schema without redundant annotations.

Titles duplicate property and definition names, and `schema_version` is a
harness default the model never needs to emit; dropping both keeps every
constraint while shrinking a schema that is sent on every call.
"""

from functools import cache
from typing import Any

from arc_agi_3.contracts.decision import CognitiveDecision


def _strip(value: Any, parent: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip(item, key)
            for key, item in value.items()
            if not (key == "title" and isinstance(item, str) and parent != "properties")
            and not (key == "schema_version" and parent == "properties")
        }
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


@cache
def decision_schema() -> dict[str, Any]:
    stripped: dict[str, Any] = _strip(CognitiveDecision.model_json_schema())
    return stripped
