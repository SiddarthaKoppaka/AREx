"""Chat boundary: role-tagged messages in, free text out.

A message may carry a PNG under "image" (bytes); only the newest one is sent.
"""

import re
from typing import Any, Protocol

THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

Message = dict[str, Any]


class ChatBackend(Protocol):
    def chat(self, messages: list[Message]) -> str: ...


def latest_image_only(messages: list[Message]) -> list[Message]:
    """Copy messages, dropping every image except the newest (saves tokens)."""
    last = max((i for i, m in enumerate(messages) if "image" in m), default=-1)
    return [
        {k: v for k, v in m.items() if k != "image" or i == last}
        for i, m in enumerate(messages)
    ]


def strip_think(text: str) -> str:
    return THINK.sub("", text).strip()
