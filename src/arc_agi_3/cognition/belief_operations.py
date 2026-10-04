"""Stage the exact belief operation selected by the LM."""

from arc_agi_3.config import BeliefConfig
from arc_agi_3.contracts.cognition import BeliefUpdate, Hypothesis
from arc_agi_3.contracts.enums import BeliefOperation

from .belief_math import update_odds

AGAINST = {BeliefOperation.WEAKEN, BeliefOperation.CONTRADICT, BeliefOperation.REJECT}


def stage_update(
    config: BeliefConfig,
    before: dict[str, Hypothesis],
    request: BeliefUpdate,
) -> tuple[dict[str, Hypothesis], set[str]]:
    target = before.get(request.hypothesis_id)
    if target is None or target.version != request.expected_version:
        raise ValueError("belief update targets an absent or stale hypothesis")
    staged = dict(before)
    touched = {target.hypothesis_id}
    staged[target.hypothesis_id] = _updated_target(config, target, request)
    if request.operation is BeliefOperation.BRANCH:
        _require_related(request, staged, minimum=2)
    if request.operation is BeliefOperation.MERGE:
        _require_related(request, staged, minimum=2)
        for related in request.related_hypothesis_ids:
            staged[related] = staged[related].model_copy(update={"status": "suspended"})
            touched.add(related)
    return staged, touched


def _updated_target(
    config: BeliefConfig, target: Hypothesis, request: BeliefUpdate
) -> Hypothesis:
    """Apply exactly the LM-requested change; no confidence is ever inferred."""
    refs = "contradicting_refs" if request.operation in AGAINST else "evidence_refs"
    updates: dict[str, object] = {
        refs: _merged(getattr(target, refs), request.evidence_refs),
        "prediction_ids": _merged(target.prediction_ids, request.prediction_ids),
    }
    numeric = {
        BeliefOperation.SUPPORT,
        BeliefOperation.WEAKEN,
        BeliefOperation.CONTRADICT,
    }
    if request.operation in numeric and request.confidence is None:
        factor = config.multiplier(request.operation.value, request.strength)
        probability = target.probability * factor
        updates["probability"] = (
            probability
            if target.belief_group
            else update_odds(target.probability, factor)
        )
    elif request.operation is BeliefOperation.REFINE:
        if not request.revised_claim:
            raise ValueError("refine requires an LM-authored revised_claim")
        updates["claim"] = request.revised_claim
    elif request.operation in {BeliefOperation.SUSPEND, BeliefOperation.BRANCH}:
        updates["status"] = "suspended"
    elif request.operation is BeliefOperation.REJECT:
        updates.update(status="rejected", probability=0.0)
    elif request.operation is BeliefOperation.ACCEPT:
        updates["status"] = "accepted"
    if request.confidence is not None:
        updates["probability"] = request.confidence
    return target.model_copy(update=updates)


def _merged(existing: tuple[str, ...], added: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*existing, *added)))


def _require_related(
    request: BeliefUpdate, staged: dict[str, Hypothesis], minimum: int
) -> None:
    if len(request.related_hypothesis_ids) < minimum:
        raise ValueError(
            f"{request.operation} requires at least {minimum} related hypotheses"
        )
    if any(key not in staged for key in request.related_hypothesis_ids):
        raise ValueError("belief relationship names an unknown hypothesis")
