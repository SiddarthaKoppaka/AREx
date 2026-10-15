"""Summarize one decision's tool_requests batch from the immutable trace.

PR-07's "batch contract": most of PR-07's gate already holds by
construction (`runtime/cognition.py` loops `decision.tool_requests`
sequentially, each isolated by `run_tool`'s own try/except, so one
failure never aborts the rest; `check_tool_requests` already rejects
duplicate `request_id`s before a decision is accepted). This module adds
the one genuinely missing piece: a typed, queryable summary of what a
batch actually did, in trace order.
"""

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.tool_batch import ToolBatchSummary


def summarize_tool_batch(
    events: list[EventEnvelope], decision_event_id: str
) -> ToolBatchSummary:
    requests = [
        event
        for event in events
        if event.event_type is EventType.TOOL_REQUEST
        and decision_event_id in event.causal_refs
    ]
    results_by_request: dict[str, EventEnvelope] = {
        event.causal_refs[0]: event
        for event in events
        if event.event_type is EventType.TOOL_RESULT and event.causal_refs
    }
    request_ids: list[str] = []
    succeeded = failed = cache_hits = 0
    for requested in requests:
        payload = requested.payload.get("request")
        request_id = payload["request_id"] if isinstance(payload, dict) else ""
        request_ids.append(str(request_id))
        result_event = results_by_request.get(requested.event_id)
        if result_event is None:
            continue
        result_payload = result_event.payload
        if result_payload.get("cache_hit"):
            cache_hits += 1
        result = result_payload.get("result")
        status = result.get("status") if isinstance(result, dict) else None
        if status == "failed":
            failed += 1
        else:
            succeeded += 1
    return ToolBatchSummary(
        decision_event_id=decision_event_id,
        requested=len(requests),
        succeeded=succeeded,
        failed=failed,
        cache_hits=cache_hits,
        request_ids_in_order=tuple(request_ids),
    )
