"""Builds `ContextStats` from `project_context`'s locals plus the extra
fields `working_set.epistemic_fields` tucked into its `epistemic` dict."""

from typing import Any

from arc_agi_3.contracts.epistemic import ContextStats
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.memory import EpisodeMemory


def context_stats(
    history: list[EventEnvelope],
    memory: EpisodeMemory | None,
    query: str,
    epistemic: dict[str, Any],
    omitted: tuple[str, ...],
    recent_count: int,
) -> ContextStats:
    """Pops its extra fields out of `epistemic` here, before
    `project_context` spreads `**epistemic` into `AgentContext`."""
    return ContextStats(
        recent_events_included=recent_count,
        history_events=len(history),
        episodic_items_included=len(memory.items) if memory else 0,
        episodic_step_ids=tuple(item.step_id for item in memory.items)
        if memory
        else (),
        episodic_query=query[:240],
        hypotheses_included=len(epistemic.get("hypotheses", ())),
        omitted_hypothesis_ids=omitted,
        consecutive_no_novelty_tool_turns=epistemic.pop("no_novelty_streak", 0),
        tool_requests_since_last_action=epistemic.pop(
            "tool_requests_since_last_action", 0
        ),
        untried_actions_from_current_state=epistemic.pop(
            "untried_actions_from_current_state", ()
        ),
    )
