"""Mutable state for a chunk spanning multiple callbacks."""

from dataclasses import dataclass

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ActionChunk

from .action_types import ActionOutcome


@dataclass
class ActiveChunk:
    chunk: ActionChunk
    decision: EventEnvelope
    started: EventEnvelope
    index: int = 0
    last: ActionOutcome | None = None
