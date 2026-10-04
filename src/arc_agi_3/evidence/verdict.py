"""Classify claim checks into one of four distinct verification states."""

from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.prediction import CHECKABLE_FIELDS, ExpectedOutcome
from arc_agi_3.contracts.transition import TransitionEvidence
from arc_agi_3.contracts.verification import (
    ClaimCheck,
    VerificationResult,
    VerificationStatus,
)
from arc_agi_3.trace.canonical import canonical_hash

from .claims import check_claims
from .transition import build_transition


def prediction_signature(expected: ExpectedOutcome) -> str | None:
    """Hash of the observable claims only, so restated predictions compare equal."""
    if not expected.checkable:
        return None
    return canonical_hash(
        expected.model_dump(mode="json", include=set(CHECKABLE_FIELDS))
    )


def summarize_checks(
    expected: ExpectedOutcome | None,
    checks: tuple[ClaimCheck, ...],
    delta: TransitionEvidence,
) -> VerificationResult:
    if expected is None:
        return VerificationResult(status="no_prediction", checked=False, delta=delta)
    mismatched = tuple(c.claim for c in checks if c.status == "mismatched")
    evaluated = any(c.status != "not_evaluable" for c in checks)
    status: VerificationStatus = (
        "mismatched" if mismatched else ("matched" if evaluated else "unchecked")
    )
    return VerificationResult(
        status=status,
        checked=evaluated,
        passed=None if not evaluated else not mismatched,
        mismatches=tuple(dict.fromkeys(mismatched)),
        checks=checks,
        prediction_id=expected.prediction_id,
        hypothesis_ids=expected.hypothesis_ids,
        prediction_signature=prediction_signature(expected),
        delta=delta,
    )


def verify_outcome(
    expected: ExpectedOutcome | None,
    before: Observation,
    after: Observation,
) -> VerificationResult:
    """Compare observable claims only; meaning of a mismatch is left to the LM."""
    delta = build_transition(before, after)
    checks = () if expected is None else check_claims(expected, before, after, delta)
    return summarize_checks(expected, checks, delta)
