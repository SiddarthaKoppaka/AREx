"""Exact-result cache for the tools whose result depends on nothing but
their own arguments - never on when they're called.

`CACHEABLE_TOOLS` is deliberately narrow: `retrieve_artifact` reads a
content-addressed, write-once artifact store (`trace/artifacts.py`), and
`simulate_world_model`/`search_world_model` look up an immutable
`WorldModel` by an exact `model_id`/`model_version` already pinned in
their own arguments (`world_model/store.py` never mutates an existing
version). All three are pure functions of (tool_name, arguments) alone,
so no event-count or other "what could have changed" signal is needed -
and none is trustworthy to invent generically. Tools whose result
genuinely depends on how much history exists (`retrieve_events`,
`retrieve_evidence`, `check_prediction_history`, `inspect_frame_region`)
or on separately-mutable workspace state (`evaluate_exploration`) are
deliberately excluded rather than cached with an unverified identity
rule. Never caches a write (`archive_artifact`,
`record_exploration_outcome`) or a model-generating call
(`consult_specialist`, `compare_specialist_reports`) - those are
excluded from `CACHEABLE_TOOLS` entirely, never bypassed automatically.
"""

from dataclasses import dataclass

from arc_agi_3.contracts.cognition import ToolResult

CACHEABLE_TOOLS = frozenset(
    {"retrieve_artifact", "simulate_world_model", "search_world_model"}
)


@dataclass(frozen=True)
class CacheEntry:
    result: ToolResult
    source_event_id: str


class ToolResultCache:
    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def lookup(self, fingerprint: str) -> CacheEntry | None:
        return self._entries.get(fingerprint)

    def store(self, fingerprint: str, result: ToolResult, source_event_id: str) -> None:
        self._entries[fingerprint] = CacheEntry(result, source_event_id)
