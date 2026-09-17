"""Reference human action efficiency arithmetic."""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def level_score(human_actions: int, agent_actions: int, cap: float = 1.15) -> float:
    if human_actions <= 0 or agent_actions <= 0:
        raise ValueError("action counts must be positive")
    return min((human_actions / agent_actions) ** 2, cap)


def environment_score(scores: tuple[float, ...], completed: int) -> float:
    if not scores:
        return 0.0
    weights = tuple(range(1, len(scores) + 1))
    total = sum(weights)
    completion_cap = sum(weights[:completed]) / total
    weighted = (
        sum(weight * score for weight, score in zip(weights, scores, strict=True))
        / total
    )
    return min(completion_cap, weighted)


def level_action_counts(events: list[EventEnvelope]) -> tuple[int, ...]:
    actions = prior_level = last_completion_actions = 0
    counts: list[int] = []
    for event in events:
        actions += event.event_type is EventType.ACTION
        if event.event_type is not EventType.OBSERVATION:
            continue
        value = event.payload.get("levels_completed", 0)
        levels = value if isinstance(value, int) else 0
        while prior_level < levels:
            counts.append(actions - last_completion_actions)
            last_completion_actions = actions
            prior_level += 1
    return tuple(counts)
