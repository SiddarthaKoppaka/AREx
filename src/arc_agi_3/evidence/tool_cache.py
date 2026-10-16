"""Exact-result cache for the tools whose result depends on nothing but
their own arguments - never on when they're called.

`retrieve_artifact` reads a content-addressed, write-once artifact store
(`trace/artifacts.py`); `simulate_world_model`/`search_world_model` look
up an immutable `WorldModel` by an exact `model_id`/`model_version`
already pinned in their own arguments (`world_model/store.py` never
mutates an existing version). `retrieve_evidence` and
`inspect_frame_region` both target one specific, already-existing event
ID: traced through `evidence_views.py`/`frame_delta.py`/`frame_region.py`
- every computation they do (including `transition_delta`'s "before"
observation lookup) is bounded by that target event's own `sequence`,
which can never be affected by events appended afterward. All five are
pure functions of (tool_name, arguments) alone, so no event-count or
other "what could have changed" signal is needed - and none is
trustworthy to invent generically.

Tools whose result genuinely depends on how much history exists right
now - `retrieve_events` (a live filter/query over the whole trace, which
can return more matches later) and `check_prediction_history` (compares
against "every past transition so far," which grows) - or on
separately-mutable workspace state (`evaluate_exploration`) are
deliberately excluded rather than cached with an unverified identity
rule. Never caches a write (`archive_artifact`,
`record_exploration_outcome`) or a model-generating call
(`consult_specialist`, `compare_specialist_reports`) - those are
excluded from `CACHEABLE_TOOLS` entirely, never bypassed automatically.
"""

from dataclasses import dataclass

from arc_agi_3.contracts.cognition import ToolResult

CACHEABLE_TOOLS = frozenset(
    {
        "retrieve_artifact",
        "retrieve_evidence",
        "inspect_frame_region",
        "simulate_world_model",
        "search_world_model",
    }
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
