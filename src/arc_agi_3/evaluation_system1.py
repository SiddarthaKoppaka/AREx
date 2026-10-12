"""System-1 (JEV) consultation counters derived mechanically from events.

Acceptance/override and progress-after-consultation metrics are
deliberately not computed here yet: they need causal correlation with
the Core Agent's next decision, which deserves its own careful design
rather than a rushed heuristic bundled into this pass. See ADR-0010.
"""

import math

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope

EMPTY_COUNTS: dict[str, float | int] = {
    "system1_requests": 0,
    "system1_suggestions": 0,
    "system1_delegated_executions": 0,
    "system1_mean_candidate_count": 0.0,
    "system1_mean_confidence": 0.0,
    "system1_mean_entropy": 0.0,
    "system1_stall_detections": 0,
    "system1_mean_latency_ms": 0.0,
}


def _entropy(probabilities: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def system1_counts(events: list[EventEnvelope]) -> dict[str, float | int]:
    decisions = [e for e in events if e.event_type is EventType.SYSTEM1_DECISION]
    if not decisions:
        return dict(EMPTY_COUNTS)
    suggestions = delegated = stalls = 0
    candidate_counts: list[float] = []
    confidences: list[float] = []
    entropies: list[float] = []
    latencies: list[float] = []
    for event in decisions:
        request = event.payload.get("request")
        result = event.payload.get("result")
        if not isinstance(request, dict) or not isinstance(result, dict):
            continue
        if request.get("execution_mode") == "suggest":
            suggestions += 1
        elif request.get("execution_mode") == "execute_one":
            delegated += 1
        candidates = request.get("candidates")
        if isinstance(candidates, list):
            candidate_counts.append(len(candidates))
        confidence = result.get("confidence")
        if isinstance(confidence, int | float):
            confidences.append(float(confidence))
        probabilities = result.get("candidate_probabilities")
        if isinstance(probabilities, list):
            values: list[float] = [
                float(probability)
                for item in probabilities
                if isinstance(item, dict)
                and isinstance((probability := item.get("probability")), int | float)
            ]
            if values:
                entropies.append(_entropy(values))
        stalled = result.get("stalled_probability")
        if isinstance(stalled, int | float) and stalled > 0.5:
            stalls += 1
        latency = result.get("latency_ms")
        if isinstance(latency, int | float):
            latencies.append(float(latency))
    return {
        "system1_requests": len(decisions),
        "system1_suggestions": suggestions,
        "system1_delegated_executions": delegated,
        "system1_mean_candidate_count": _mean(candidate_counts),
        "system1_mean_confidence": _mean(confidences),
        "system1_mean_entropy": _mean(entropies),
        "system1_stall_detections": stalls,
        "system1_mean_latency_ms": _mean(latencies),
    }
