"""Deterministic episodic summaries for older canonical events."""

from collections import defaultdict

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.memory import EpisodeMemory, EpisodeMemoryItem

from .episode_summary import describe
from .projection import compact_context_event

_SALIENT = {
    EventType.OBSERVATION,
    EventType.MODEL_DECISION,
    EventType.TOOL_RESULT,
    EventType.ACTION,
    EventType.TRANSITION,
    EventType.VERIFICATION,
    EventType.INTERRUPT,
    EventType.RECOVERY,
    EventType.BRANCH_FROZEN,
    EventType.BRANCH_FORKED,
    EventType.FAILURE,
}


def summarize_episode(
    events: list[EventEnvelope], item_limit: int = 8, query: str = ""
) -> EpisodeMemory | None:
    """Summarize the latest salient older steps without changing source events."""
    if item_limit <= 0:
        return None
    salient = [event for event in events if event.event_type in _SALIENT]
    if not salient:
        return None
    grouped: dict[int, list[EventEnvelope]] = defaultdict(list)
    for event in salient:
        grouped[event.step_id].append(event)
    candidates = [_item(step, grouped[step]) for step in sorted(grouped)]
    terms = {term.lower() for term in query.split() if len(term) > 2}

    def score(item: EpisodeMemoryItem) -> tuple[int, int]:
        text = " ".join(
            (item.summary, *item.facts_learned, *item.failed_approaches)
        ).lower()
        return sum(term in text for term in terms), item.step_id

    items = tuple(
        sorted(sorted(candidates, key=score)[-item_limit:], key=lambda x: x.step_id)
    )
    summarized = sum(len(item.source_event_refs) for item in items)
    return EpisodeMemory(
        source_event_count=len(events),
        summarized_event_count=summarized,
        omitted_event_count=len(events) - summarized,
        items=items,
    )


def _item(step: int, events: list[EventEnvelope]) -> EpisodeMemoryItem:
    ordered = sorted(events, key=lambda event: event.sequence)
    summary, facts, failures = describe(ordered)
    visible = ordered[-6:]
    refs = tuple(event.event_id for event in visible)
    return EpisodeMemoryItem(
        step_id=step,
        turn_range=(step, step),
        first_sequence=ordered[0].sequence,
        last_sequence=ordered[-1].sequence,
        summary=summary,
        facts_learned=facts,
        failed_approaches=failures,
        evidence_refs=refs,
        source_event_refs=refs,
        events=tuple(compact_context_event(event) for event in visible),
    )
