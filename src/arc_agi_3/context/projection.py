"""Bounded event summaries for automatic model-facing context only."""

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.retrieval import ContextEvent

from .retrieved_projection import retrieved_evidence


def _bounded(value: JsonValue, depth: int = 0) -> JsonValue:
    if depth >= 5:
        return "[omitted]"
    if isinstance(value, str):
        return value[:240] + ("…" if len(value) > 240 else "")
    if isinstance(value, list):
        return [_bounded(item, depth + 1) for item in value[:8]]
    if isinstance(value, dict):
        return {
            key: _bounded(item, depth + 1)
            for key, item in list(sorted(value.items()))[:16]
            if key not in {"frame", "manifest", "decision", "versions"}
        }
    return value


def _payload(event: EventEnvelope) -> dict[str, JsonValue]:
    source = event.payload
    if event.event_type is EventType.OBSERVATION:
        keys = (
            "game_id",
            "observation_hash",
            "state",
            "levels_completed",
            "win_levels",
            "available_actions",
            "guid",
            "full_reset",
        )
        return {key: source[key] for key in keys if key in source}
    if event.event_type is EventType.MODEL_DECISION:
        decision = source.get("decision")
        if isinstance(decision, dict):
            return {
                key: _bounded(decision[key])
                for key in (
                    "mode",
                    "assessment",
                    "intent",
                    "considered_options",
                    "decision_summary",
                    "observation_summary",
                    "expected_result",
                    "evidence_refs",
                )
                if key in decision
            }
        return {}
    if event.event_type is EventType.RUN_STARTED:
        return {}
    if event.event_type is EventType.TOOL_RESULT:
        retrieved = retrieved_evidence(source)
        if retrieved is not None:
            return {"result": retrieved}
    if event.event_type in {
        EventType.HYPOTHESIS,
        EventType.TASK_UPDATE,
        EventType.WORLD_MODEL,
        EventType.SCRATCHPAD_UPDATE,
    }:
        return {}
    return {
        key: _bounded(value)
        for key, value in sorted(source.items())[:16]
        if key != "frame"
    }


def compact_context_event(event: EventEnvelope) -> ContextEvent:
    """Keep provenance and useful deltas without repeated frames or large payloads."""
    return ContextEvent(
        event_id=event.event_id,
        event_hash=event.event_hash,
        sequence=event.sequence,
        step_id=event.step_id,
        event_type=event.event_type,
        component=event.component,
        causal_refs=event.causal_refs,
        payload=_payload(event),
    )
