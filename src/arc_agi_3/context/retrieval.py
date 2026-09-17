"""Exact event retrieval; query intent is supplied by the LM."""

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.retrieval import ContextEvent, RetrievalQuery, RetrievalResult
from arc_agi_3.trace.canonical import canonical_json


def context_event(event: EventEnvelope) -> ContextEvent:
    return ContextEvent(
        event_id=event.event_id,
        event_hash=event.event_hash,
        sequence=event.sequence,
        event_type=event.event_type,
        component=event.component,
        causal_refs=event.causal_refs,
        payload=event.payload,
    )


class EventRetriever:
    def __init__(self, events: list[EventEnvelope]) -> None:
        self._events = tuple(events)
        self._by_id = {event.event_id: event for event in events}

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        events = self._candidate_events(query)
        matches = [event for event in events if self._matches(event, query)]
        if query.order == "descending":
            matches.reverse()
        return RetrievalResult(
            query=query,
            matches=tuple(context_event(event) for event in matches[: query.limit]),
        )

    def _candidate_events(self, query: RetrievalQuery) -> list[EventEnvelope]:
        if not query.event_ids:
            return list(self._events)
        return [
            self._by_id[event_id]
            for event_id in query.event_ids
            if event_id in self._by_id
        ]

    @staticmethod
    def _matches(event: EventEnvelope, query: RetrievalQuery) -> bool:
        if query.event_types and event.event_type not in query.event_types:
            return False
        if query.causal_ref and query.causal_ref not in event.causal_refs:
            return False
        if query.component and event.component != query.component:
            return False
        if query.min_sequence is not None and event.sequence < query.min_sequence:
            return False
        if query.max_sequence is not None and event.sequence > query.max_sequence:
            return False
        return not query.literal_text or query.literal_text in canonical_json(
            event.payload
        )
