"""Tool-request repeat counters derived mechanically from immutable events.

Mirrors the environment-action repeat machinery (evaluation_epistemic.py's
repeated_experiments/unjustified_repeats) for cognitive tool requests.
"""

from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.evidence import tool_fingerprint


def tool_repeat_counts(events: list[EventEnvelope]) -> dict[str, int]:
    seen: dict[str, str | None] = {}
    active_observation: str | None = None
    total = unique = repeated = unjustified = 0
    novelty_by_turn: dict[int, bool] = {}
    for event in events:
        if event.event_type is EventType.OBSERVATION:
            active_observation = event.event_id
            continue
        if event.event_type is not EventType.TOOL_REQUEST:
            continue
        payload = event.payload.get("request")
        if not isinstance(payload, dict):
            continue
        fingerprint = tool_fingerprint(ToolRequest.model_validate(payload))
        total += 1
        novel = fingerprint not in seen or seen[fingerprint] != active_observation
        if fingerprint not in seen:
            unique += 1
        else:
            repeated += 1
            unjustified += int(not novel)
        novelty_by_turn[event.step_id] = (
            novelty_by_turn.get(event.step_id, False) or novel
        )
        seen[fingerprint] = active_observation
    streak = longest = 0
    for step in sorted(novelty_by_turn):
        streak = 0 if novelty_by_turn[step] else streak + 1
        longest = max(longest, streak)
    return {
        "tool_requests": total,
        "unique_tool_requests": unique,
        "repeated_tool_requests": repeated,
        "unjustified_tool_repeats": unjustified,
        "consecutive_no_novelty_turns": longest,
    }
