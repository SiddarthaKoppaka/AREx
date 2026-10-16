"""Tool-request repeat counters derived mechanically from immutable events.

Mirrors the environment-action repeat machinery (evaluation_epistemic.py's
repeated_experiments/unjustified_repeats) for cognitive tool requests.
"""

from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.evidence import tool_fingerprint


def _novelty_pass(
    events: list[EventEnvelope],
) -> tuple[int, int, int, int, dict[int, bool]]:
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
    return total, unique, repeated, unjustified, novelty_by_turn


def _streaks(novelty_by_turn: dict[int, bool]) -> tuple[int, int]:
    """Returns (current trailing streak, longest streak ever)."""
    streak = longest = 0
    for step in sorted(novelty_by_turn):
        streak = 0 if novelty_by_turn[step] else streak + 1
        longest = max(longest, streak)
    return streak, longest


def current_no_novelty_streak(events: list[EventEnvelope]) -> int:
    """How many of the most recent consecutive turns produced zero novel
    tool request, as of right now - read live into `AgentContext`
    (`ContextStats.consecutive_no_novelty_tool_turns`) so the model can see
    it; nothing acts on this number automatically."""
    _, _, _, _, novelty_by_turn = _novelty_pass(events)
    current, _ = _streaks(novelty_by_turn)
    return current


def tool_requests_since_last_action(events: list[EventEnvelope]) -> int:
    """Consecutive tool requests since the last real environment action, as
    of right now - read live into `ContextStats.tool_requests_since_last_
    action` so the model can see its own tool-eagerness; nothing acts on
    this number automatically. 0 at episode start and immediately after an
    action."""
    count = 0
    for event in reversed(events):
        if event.event_type is EventType.ACTION:
            break
        if event.event_type is EventType.TOOL_REQUEST:
            count += 1
    return count


def tool_repeat_counts(events: list[EventEnvelope]) -> dict[str, int]:
    total, unique, repeated, unjustified, novelty_by_turn = _novelty_pass(events)
    _, longest = _streaks(novelty_by_turn)
    return {
        "tool_requests": total,
        "unique_tool_requests": unique,
        "repeated_tool_requests": repeated,
        "unjustified_tool_repeats": unjustified,
        "consecutive_no_novelty_turns": longest,
    }
