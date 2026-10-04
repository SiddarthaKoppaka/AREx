"""Hypothesis ledger: link model-owned hypotheses to recorded prediction tests.

Read-only over the trace. Confidence and status are never touched; a mismatch
the model has not yet cited stays listed as an unacknowledged contradiction.
"""

from arc_agi_3.contracts.cognition import Hypothesis
from arc_agi_3.contracts.epistemic import HypothesisLedgerEntry, PredictionTest

from .history import TransitionRecord
from .targets import targeted

LIVE = {"active": 0, "accepted": 1, "suspended": 2}


def _test(record: TransitionRecord) -> PredictionTest | None:
    event = record.verification_event
    if event is None or record.verification_status is None:
        return None
    payload = event.payload
    mismatches = payload.get("mismatches")
    prediction_id = payload.get("prediction_id")
    return PredictionTest(
        verification_event_id=event.event_id,
        step_id=event.step_id,
        prediction_id=prediction_id if isinstance(prediction_id, str) else None,
        status=record.verification_status,
        mismatches=tuple(map(str, mismatches)) if isinstance(mismatches, list) else (),
    )


def hypothesis_ledger(
    hypotheses: tuple[Hypothesis, ...],
    records: list[TransitionRecord],
    targets: dict[str, tuple[str, ...]],
    *,
    limit: int = 8,
    test_limit: int = 3,
) -> tuple[tuple[HypothesisLedgerEntry, ...], tuple[str, ...]]:
    """Return (included entries, omitted hypothesis IDs)."""
    entries: list[HypothesisLedgerEntry] = []
    for hypothesis in hypotheses:
        tests = [
            test
            for record in records
            if hypothesis.hypothesis_id in targeted(record, targets)
            and (test := _test(record)) is not None
        ]
        cited = {*hypothesis.evidence_refs, *hypothesis.contradicting_refs}
        open_ids = tuple(
            test.verification_event_id
            for test in tests
            if test.status == "mismatched" and test.verification_event_id not in cited
        )
        entries.append(
            HypothesisLedgerEntry(
                hypothesis=hypothesis,
                tests=tuple(tests[-test_limit:]),
                omitted_tests=max(0, len(tests) - test_limit),
                unacknowledged_contradictions=open_ids[-test_limit:],
                unacknowledged_total=len(open_ids),
            )
        )
    live = sorted(
        (item for item in entries if item.hypothesis.status in LIVE),
        key=lambda item: (
            -item.unacknowledged_total,
            LIVE[item.hypothesis.status],
            -item.hypothesis.probability,
            item.hypothesis.hypothesis_id,
        ),
    )
    included = tuple(live[:limit])
    shown = {item.hypothesis.hypothesis_id for item in included}
    omitted = tuple(h.hypothesis_id for h in hypotheses if h.hypothesis_id not in shown)
    return included, omitted


def unresolved_contradictions(
    entries: tuple[HypothesisLedgerEntry, ...],
) -> tuple[str, ...]:
    """One line per hypothesis with mismatches it does not yet cite."""
    return tuple(
        f"{item.hypothesis.hypothesis_id}: {item.unacknowledged_total} prediction "
        f"mismatch(es) not cited in evidence_refs or contradicting_refs; latest "
        f"{item.unacknowledged_contradictions[-1]}"
        for item in entries
        if item.unacknowledged_total
    )
