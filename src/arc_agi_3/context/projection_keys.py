"""Payload keys kept when an event is summarized for automatic context.

Exact payloads remain retrievable by event ID; these are views, not rewrites.
"""

from arc_agi_3.contracts.enums import EventType

SUMMARY_KEYS: dict[EventType, tuple[str, ...]] = {
    EventType.OBSERVATION: (
        "game_id",
        "observation_hash",
        "state",
        "levels_completed",
        "win_levels",
        "available_actions",
        "guid",
        "full_reset",
    ),
    EventType.TRANSITION: (
        "changed_cells",
        "metadata_changes",
        "changed_regions",
        "translations",
        "omitted_translations",
    ),
    EventType.VERIFICATION: (
        "status",
        "mismatches",
        "prediction_id",
        "hypothesis_ids",
    ),
    EventType.EXPERIMENT: ("experiment", "repeat", "unjustified_repeat"),
}

HIDDEN_IN_CONTEXT = frozenset(
    {
        EventType.BUDGET,
        EventType.RUN_STARTED,
        EventType.HYPOTHESIS,
        EventType.BELIEF_UPDATE,
        EventType.TASK_UPDATE,
        EventType.WORLD_MODEL,
        EventType.SCRATCHPAD_UPDATE,
        EventType.SPECIALIST_REPORT,
        EventType.SPECIALIST_REVIEW,
    }
)
