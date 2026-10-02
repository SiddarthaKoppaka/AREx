"""Epistemic-loop counters derived mechanically from immutable events."""

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def verification_status(payload: dict[str, JsonValue]) -> str:
    """Map current and legacy (checked/passed only) verification payloads."""
    status = payload.get("status")
    if isinstance(status, str):
        return status
    if payload.get("checked") is True:
        return "matched" if payload.get("passed") is True else "mismatched"
    return "no_prediction"


def epistemic_counts(events: list[EventEnvelope]) -> dict[str, int]:
    statuses = [
        verification_status(event.payload)
        for event in events
        if event.event_type is EventType.VERIFICATION
    ]
    experiments = [e for e in events if e.event_type is EventType.EXPERIMENT]
    repeats = [e.payload.get("repeat") for e in experiments]
    return {
        "prediction_mismatches": statuses.count("mismatched"),
        "predictions_matched": statuses.count("matched"),
        "predictions_unchecked": statuses.count("unchecked"),
        "actions_without_prediction": statuses.count("no_prediction"),
        "experiments": sum(
            e.payload.get("experiment") is not None for e in experiments
        ),
        "repeated_experiments": sum(
            isinstance(item, dict) and bool(item.get("equivalent_prior"))
            for item in repeats
        ),
        "unjustified_repeats": sum(
            e.payload.get("unjustified_repeat") is True for e in experiments
        ),
        "hypothesis_revisions": sum(
            e.event_type is EventType.BELIEF_UPDATE for e in events
        ),
        "plan_deviations": sum(
            e.event_type is EventType.ACTION
            and e.payload.get("matches_planned_next_test") is False
            for e in events
        ),
    }
