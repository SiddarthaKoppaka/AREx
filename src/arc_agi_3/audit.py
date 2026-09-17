"""Mechanical audit for the LM/harness agency boundary."""

from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.events import EventEnvelope


def audit_agency_boundary(events: list[EventEnvelope]) -> tuple[str, ...]:
    by_id = {event.event_id: event for event in events}
    violations: list[str] = []
    actions_by_decision: dict[str, list[object]] = {}
    for action_event in (
        event for event in events if event.event_type is EventType.ACTION
    ):
        parents = [by_id.get(reference) for reference in action_event.causal_refs]
        decisions = [
            event
            for event in parents
            if event and event.event_type is EventType.MODEL_DECISION
        ]
        if len(decisions) != 1:
            violations.append(f"{action_event.event_id}: missing unique LM decision")
            continue
        value = decisions[0].payload.get("decision")
        if not isinstance(value, dict):
            violations.append(f"{action_event.event_id}: malformed LM decision")
            continue
        if value.get("mode") != DecisionMode.EXECUTE:
            violations.append(f"{action_event.event_id}: non-execute decision")
        actions_by_decision.setdefault(decisions[0].event_id, []).append(
            action_event.payload.get("action")
        )
    for decision_id, actual in actions_by_decision.items():
        value = by_id[decision_id].payload["decision"]
        if not isinstance(value, dict):
            continue
        direct = value.get("action")
        chunk = value.get("action_chunk")
        authorized = [direct] if direct is not None else _chunk_actions(chunk)
        if actual != authorized[: len(actual)]:
            violations.append(f"{decision_id}: action sequence was substituted")
    lm_owned = {
        EventType.HYPOTHESIS,
        EventType.BELIEF_UPDATE,
        EventType.TASK_UPDATE,
        EventType.WORLD_MODEL,
        EventType.TOOL_REQUEST,
        EventType.CHUNK_STARTED,
        EventType.CHUNK_FINISHED,
        EventType.INTERRUPT,
    }
    for event in (item for item in events if item.event_type in lm_owned):
        parents = [by_id.get(reference) for reference in event.causal_refs]
        if not any(
            parent and parent.event_type is EventType.MODEL_DECISION
            for parent in parents
        ):
            violations.append(f"{event.event_id}: mutation lacks LM decision")
    for event in (item for item in events if item.event_type is EventType.TOOL_RESULT):
        parents = [by_id.get(reference) for reference in event.causal_refs]
        if not any(
            parent and parent.event_type is EventType.TOOL_REQUEST for parent in parents
        ):
            violations.append(f"{event.event_id}: tool result lacks request")
    requested_recoveries = [
        item
        for item in events
        if item.event_type is EventType.RECOVERY
        and item.payload.get("status") == "requested"
    ]
    for event in requested_recoveries:
        parents = [by_id.get(reference) for reference in event.causal_refs]
        if not any(
            parent and parent.event_type is EventType.MODEL_DECISION
            for parent in parents
        ):
            violations.append(f"{event.event_id}: recovery lacks LM decision")
    branch_events = {EventType.BRANCH_FROZEN, EventType.BRANCH_FORKED}
    for event in (item for item in events if item.event_type in branch_events):
        parents = [by_id.get(reference) for reference in event.causal_refs]
        if not any(
            parent and parent.event_type is EventType.RECOVERY for parent in parents
        ):
            violations.append(
                f"{event.event_id}: branch mutation lacks recovery request"
            )
    return tuple(violations)


def _chunk_actions(value: object) -> list[object]:
    if not isinstance(value, dict):
        return []
    steps = value.get("steps")
    if not isinstance(steps, list):
        return []
    return [step.get("action") for step in steps if isinstance(step, dict)]
