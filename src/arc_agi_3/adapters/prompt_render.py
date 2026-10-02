"""Display-only prompt serialization; canonical traces and hashes are unaffected.

Drops `schema_version`, trace-integrity `event_hash`, and null/empty-valued
dict keys, none of which carry information for the model (absent and default
are equivalent in every schema; hashes stay verifiable in the trace).
"""

from typing import Any

from pydantic import BaseModel

from arc_agi_3.trace.canonical import canonical_json

_EMPTY: tuple[Any, ...] = (None, [], {}, ())
_HIDDEN = frozenset({"schema_version", "event_hash"})


def _strip(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _strip(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            key: _strip(item)
            for key, item in value.items()
            if key not in _HIDDEN and not any(item == e for e in _EMPTY)
        }
    if isinstance(value, list | tuple):
        return [_strip(item) for item in value]
    return value


def render(value: Any) -> str:
    return canonical_json(_strip(value))
