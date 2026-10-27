"""Context eviction: keep the system prompt and the most recent messages."""

from .chat import Message

IMAGE_COST = 3000  # rough char-equivalent of one rendered frame


def _cost(message: Message) -> int:
    return len(message["content"]) + (IMAGE_COST if "image" in message else 0)


def evict(messages: list[Message], budget_chars: int) -> list[Message]:
    """Drop the oldest non-system messages until the rest fit the budget.

    The first kept message after the system prompt is always a user turn.
    """
    system, rest = messages[:1], messages[1:]
    used = sum(_cost(m) for m in system)
    kept: list[Message] = []
    for message in reversed(rest):
        used += _cost(message)
        if used > budget_chars and kept:
            break
        kept.append(message)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return system + kept
