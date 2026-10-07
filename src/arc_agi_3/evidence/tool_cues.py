"""Per-tool-request evidence table: what each distinct cognitive tool call
has already been asked, from which observation. Mirrors action_cues.py.

Mechanical only: it never decides a request is a "wasted" repeat, only
reports attempts and whether the observation has changed since the last
one - the Core Agent judges what that means.
"""

from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.epistemic import ToolRequestEvidence
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.trace.canonical import canonical_hash


def tool_fingerprint(request: ToolRequest) -> str:
    return canonical_hash(
        {"tool_name": request.tool_name, "arguments": request.arguments}
    )


def tool_request_evidence(
    history: list[EventEnvelope],
    current_observation_event_id: str | None,
    *,
    limit: int = 8,
) -> tuple[ToolRequestEvidence, ...]:
    groups: dict[str, list[EventEnvelope]] = {}
    requests: dict[str, ToolRequest] = {}
    observation_at: dict[str, str | None] = {}
    active_observation: str | None = None
    for event in history:
        if event.event_type is EventType.OBSERVATION:
            active_observation = event.event_id
            continue
        if event.event_type is not EventType.TOOL_REQUEST:
            continue
        payload = event.payload.get("request")
        if not isinstance(payload, dict):
            continue
        request = ToolRequest.model_validate(payload)
        key = tool_fingerprint(request)
        groups.setdefault(key, []).append(event)
        requests.setdefault(key, request)
        observation_at[event.event_id] = active_observation
    table: list[ToolRequestEvidence] = []
    for key, events in list(groups.items())[:limit]:
        request = requests[key]
        from_current = sum(
            observation_at.get(e.event_id) == current_observation_event_id
            for e in events
        )
        table.append(
            ToolRequestEvidence(
                tool_name=request.tool_name,
                arguments=request.arguments,
                fingerprint=key,
                attempts=len(events),
                attempts_from_current_state=from_current,
                last_request_event_id=events[-1].event_id,
            )
        )
    return tuple(table)
