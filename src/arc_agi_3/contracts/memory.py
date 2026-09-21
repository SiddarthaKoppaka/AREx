"""Auditable summaries of events older than the recent context window."""

from pydantic import Field

from .base import Contract
from .retrieval import ContextEvent


class EpisodeMemoryItem(Contract):
    step_id: int = Field(ge=0)
    turn_range: tuple[int, int]
    first_sequence: int = Field(ge=0)
    last_sequence: int = Field(ge=0)
    summary: str
    facts_learned: tuple[str, ...] = ()
    failed_approaches: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...]
    source_event_refs: tuple[str, ...]
    events: tuple[ContextEvent, ...]


class EpisodeMemory(Contract):
    source_event_count: int = Field(ge=0)
    summarized_event_count: int = Field(ge=0)
    omitted_event_count: int = Field(ge=0)
    items: tuple[EpisodeMemoryItem, ...]
