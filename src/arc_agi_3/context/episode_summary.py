"""Concise public text derived mechanically from one turn's events."""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope


def describe(
    events: list[EventEnvelope],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    summaries: list[str] = []
    facts: list[str] = []
    failures: list[str] = []
    for event in events:
        payload = event.payload
        if event.event_type is EventType.ACTION:
            action = payload.get("action")
            if isinstance(action, dict):
                summaries.append(
                    f"Executed action {action.get('action_id', 'unknown')}."
                )
        elif event.event_type is EventType.TRANSITION:
            changed = payload.get("changed_cells")
            facts.append(f"Transition changed {changed} cells.")
        elif event.event_type is EventType.VERIFICATION:
            if payload.get("passed") is False:
                failures.append("Expected outcome verification failed.")
        elif event.event_type is EventType.RECOVERY:
            if payload.get("status") == "failed":
                failures.append(f"Recovery failed: {payload.get('reason', 'unknown')}.")
        elif event.event_type is EventType.MODEL_DECISION:
            decision = payload.get("decision")
            if isinstance(decision, dict):
                summary = decision.get("decision_summary")
                if isinstance(summary, str):
                    summaries.append(summary)
        elif event.event_type is EventType.OBSERVATION:
            state = payload.get("state")
            facts.append(
                f"Observed state {state} at {payload.get('observation_hash')}."
            )
    summary = " ".join(summaries[:3]) or "Recorded environment evidence."
    return summary, tuple(facts[:4]), tuple(failures[:3])
