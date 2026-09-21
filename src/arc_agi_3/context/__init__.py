"""Deterministic, provenance-preserving context projection."""

from .episodic import summarize_episode
from .projection import compact_context_event
from .retrieval import EventRetriever, context_event

__all__ = [
    "EventRetriever",
    "compact_context_event",
    "context_event",
    "summarize_episode",
]
