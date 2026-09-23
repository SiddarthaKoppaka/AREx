"""Expose bounded exact retrieval evidence from a requested tool result."""

from pydantic import JsonValue


def retrieved_evidence(source: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    result = source.get("result")
    if not isinstance(result, dict):
        return None
    output = result.get("output")
    if not isinstance(output, dict):
        return None
    evidence = output.get("evidence")
    if isinstance(evidence, dict):
        return {"status": result.get("status"), "evidence": evidence}
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
            if size <= 4096:
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
