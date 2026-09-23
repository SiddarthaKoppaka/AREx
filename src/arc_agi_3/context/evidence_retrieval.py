"""Budgeted, purpose-labelled views over immutable event history."""

from typing import Any, Literal

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.evidence import EvidenceQuery
from arc_agi_3.trace.canonical import canonical_json

from .evidence_views import view_event


def retrieve_evidence(
    history: list[EventEnvelope], query: EvidenceQuery
) -> tuple[Literal["complete", "partial"], dict[str, Any], tuple[str, ...]]:
    by_id = {event.event_id: event for event in history}
    missing = [event_id for event_id in query.event_ids if event_id not in by_id]
    if missing:
        raise ValueError(f"unknown evidence event IDs: {missing}")
    items: list[dict[str, Any]] = []
    omitted: list[str] = []
    for event_id in query.event_ids:
        item = view_event(by_id[event_id], history, query)
        candidate = _output(query, [*items, item], omitted)
        if _serialized_size(candidate) > query.token_budget:
            omitted.append(event_id)
        else:
            items.append(item)
    output = _output(query, items, omitted)
    while items and _serialized_size(output) > query.token_budget:
        omitted.insert(0, items.pop()["event_id"])
        output = _output(query, items, omitted)
    if _serialized_size(output) > query.token_budget:
        raise ValueError("evidence metadata exceeds requested token budget")
    status: Literal["complete", "partial"] = "partial" if omitted else "complete"
    return status, output, tuple(item["event_id"] for item in items)


def _output(
    query: EvidenceQuery, items: list[dict[str, Any]], omitted: list[str]
) -> dict[str, Any]:
    return {
        "view": query.view,
        "purpose": query.purpose,
        "token_budget": query.token_budget,
        "items": items,
        "omitted_event_ids": omitted,
        "budget_method": "serialized_utf8_bytes_upper_bound",
    }


def _serialized_size(value: object) -> int:
    return len(canonical_json(value).encode("utf-8"))
