"""Mechanically derived research counters."""

from arc_agi_3.contracts.enums import EventType, ExecutionStatus, Resource
from arc_agi_3.contracts.events import EventEnvelope


def resource_spend(events: list[EventEnvelope], resource: Resource) -> int:
    return sum(
        amount
        for event in events
        if event.event_type is EventType.BUDGET
        and event.payload.get("resource") == resource
        and isinstance((amount := event.payload.get("amount")), int)
    )


def repeated_actions(events: list[EventEnvelope]) -> int:
    actions = [
        event.payload.get("action")
        for event in events
        if event.event_type is EventType.ACTION
    ]
    return sum(left == right for left, right in zip(actions, actions[1:], strict=False))


def event_research_counts(events: list[EventEnvelope]) -> dict[str, int]:
    interrupts = [e for e in events if e.event_type is EventType.INTERRUPT]
    recoveries = [e for e in events if e.event_type is EventType.RECOVERY]
    return {
        "search_node_expansions": resource_spend(events, Resource.SEARCH_NODES),
        "simulations": resource_spend(events, Resource.SIMULATIONS),
        "wall_time_ms": resource_spend(events, Resource.WALL_TIME_MS),
        "prediction_mismatches": sum(
            e.event_type is EventType.VERIFICATION
            and not bool(e.payload.get("passed", False))
            for e in events
        ),
        "soft_interrupts": sum(
            e.payload.get("status") == ExecutionStatus.SOFT_INTERRUPT
            for e in interrupts
        ),
        "hard_stops": sum(
            e.payload.get("status") == ExecutionStatus.HARD_STOP for e in interrupts
        ),
        "recovery_attempts": len(recoveries),
        "recovery_failures": sum(
            e.payload.get("status") == "failed" for e in recoveries
        ),
        "branch_forks": sum(e.event_type is EventType.BRANCH_FORKED for e in events),
        "repeated_actions": repeated_actions(events),
        "world_model_versions": _world_model_versions(events),
    }


def _world_model_versions(events: list[EventEnvelope]) -> int:
    total = 0
    for event in events:
        versions = event.payload.get("versions", [])
        if event.event_type is EventType.WORLD_MODEL and isinstance(versions, list):
            total += len(versions)
    return total
