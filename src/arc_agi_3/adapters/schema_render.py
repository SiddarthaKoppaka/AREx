"""Strip redundant annotations from a model's JSON schema before showing it.

Titles duplicate property and definition names, and `schema_version` is a
harness default no model ever needs to emit; dropping both shrinks a schema
that is sent on every call without losing any constraint.
"""

from typing import Any


def strip_schema(value: Any, parent: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_schema(item, key)
            for key, item in value.items()
            if not (key == "title" and isinstance(item, str) and parent != "properties")
            and not (key == "schema_version" and parent == "properties")
        }
    if isinstance(value, list):
        return [strip_schema(item) for item in value]
    return value
