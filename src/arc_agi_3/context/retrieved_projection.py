"""Expose bounded exact retrieval evidence from a requested tool result.

The exact result is always preserved verbatim in `events.jsonl` and stays
retrievable by event ID — this module only decides what gets replayed into
*automatic*, every-turn context. A model that explicitly asked for
`view=full` still gets the full frame back from the tool call itself; it
is only the next turn's automatic context that stays capacity-safe.
"""

from pydantic import JsonValue

MAX_ITEM_CHARACTERS = 4096


def _bounded_item(item: dict[str, JsonValue]) -> dict[str, JsonValue]:
    size = len(str(item.get("content")))
    if size <= MAX_ITEM_CHARACTERS:
        return item
    return {
        "event_id": item.get("event_id"),
        "event_hash": item.get("event_hash"),
        "event_type": item.get("event_type"),
        "view": item.get("view"),
        "content_projection": {
            "omitted_from_automatic_context": True,
            "size_characters": size,
            "reason": "large exact evidence",
            "suggested_views": ["rle_frame", "inspect_frame_region"],
        },
    }


def _bounded_evidence(evidence: dict[str, JsonValue]) -> dict[str, JsonValue]:
    items = evidence.get("items")
    if not isinstance(items, list):
        return evidence
    bounded: list[JsonValue] = [
        _bounded_item(item) if isinstance(item, dict) else item for item in items
    ]
    return {**evidence, "items": bounded}


def retrieved_evidence(source: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    result = source.get("result")
    if not isinstance(result, dict):
        return None
    output = result.get("output")
    if not isinstance(output, dict):
        return None
    evidence = output.get("evidence")
    if isinstance(evidence, dict):
        return {
            "status": result.get("status"),
            "evidence": _bounded_evidence(evidence),
        }
    if isinstance(output.get("retrodiction"), dict):
        return {"status": result.get("status"), "retrodiction": output["retrodiction"]}
    if "artifact_ref" in output and "content" in output:
        return {"status": result.get("status"), "artifact": output}
    retrieval = output.get("retrieval")
    if not isinstance(retrieval, dict):
        return None
    matches = retrieval.get("matches")
    if not isinstance(matches, list):
        return None
    projected: list[JsonValue] = []
    for match in matches[:2]:
        if not isinstance(match, dict):
            continue
        payload = match.get("payload")
        if not isinstance(payload, dict):
            continue
        frame = payload.get("frame")
        visible: dict[str, JsonValue] = {
            key: payload[key] for key in ("observation_hash", "state") if key in payload
        }
        if isinstance(frame, list):
            size = len(str(frame))
            if size <= MAX_ITEM_CHARACTERS:
                visible["frame"] = frame
            else:
                visible["frame_size_characters"] = size
        projected.append(
            {
                "event_id": match.get("event_id"),
                "event_type": match.get("event_type"),
                "payload": visible,
            }
        )
    return {
        "status": result.get("status"),
        "retrieved_matches": projected,
        "match_count": len(matches),
    }
