"""Model-facing view of episodic memory: summaries plus retrievable refs only."""

from pydantic import JsonValue

from arc_agi_3.contracts.memory import EpisodeMemory


def episode_prompt_view(memory: EpisodeMemory | None) -> dict[str, JsonValue] | None:
    if memory is None:
        return None
    return {
        "source_event_count": memory.source_event_count,
        "omitted_event_count": memory.omitted_event_count,
        "items": [
            {
                "step_id": item.step_id,
                "summary": item.summary,
                "facts_learned": list(item.facts_learned),
                "failed_approaches": list(item.failed_approaches),
                "evidence_refs": list(item.evidence_refs),
            }
            for item in memory.items
        ],
    }
