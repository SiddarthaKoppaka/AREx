"""The four verification states stay distinct and only observables are checked."""

import pytest
from pydantic import ValidationError

from arc_agi_3.contracts.prediction import (
    CellExpectation,
    ExpectedOutcome,
    TranslationExpectation,
)
from arc_agi_3.contracts.verification import VerificationResult
from arc_agi_3.evaluation_epistemic import verification_status
from arc_agi_3.testing.epistemic import grid
from arc_agi_3.verification import compare_observations, verify_outcome


def frames() -> tuple[list[list[int]], list[list[int]]]:
    before, after = [[0] * 5 for _ in range(5)], [[0] * 5 for _ in range(5)]
    before[2][1] = after[2][2] = 1
    return before, after


def test_prediction_checked_and_matched() -> None:
    before, after = frames()
    expected = ExpectedOutcome(
        prediction_id="p1",
        hypothesis_ids=("h1",),
        translations=(TranslationExpectation(value=1, d_row=0, d_col=1),),
        max_changed_cells=2,
    )
    result = verify_outcome(expected, grid(before), grid(after))
    assert (result.status, result.checked, result.passed) == ("matched", True, True)
    assert result.prediction_id == "p1" and result.hypothesis_ids == ("h1",)
    assert {check.status for check in result.checks} == {"matched"}


def test_prediction_checked_and_mismatched_with_structured_evidence() -> None:
    before, after = frames()
    expected = ExpectedOutcome(
        translations=(TranslationExpectation(value=1, d_row=-1, d_col=0),),
        no_op=True,
    )
    result = verify_outcome(expected, grid(before), grid(after))
    assert (result.status, result.checked, result.passed) == (
        "mismatched",
        True,
        False,
    )
    assert result.mismatches == ("no_op", "translation")
    translation = next(c for c in result.checks if c.claim == "translation")
    assert translation.observed == [{"from": [2, 1], "delta": [0, 1]}]


def test_prediction_that_cannot_be_checked_is_not_passed() -> None:
    before, after = frames()
    prose = ExpectedOutcome(description="Something interesting should happen.")
    result = verify_outcome(prose, grid(before), grid(after))
    assert (result.status, result.checked, result.passed) == ("unchecked", False, None)
    outside = ExpectedOutcome(cell_values=(CellExpectation(row=40, col=40, value=1),))
    result = verify_outcome(outside, grid(before), grid(after))
    assert result.status == "unchecked" and result.passed is None
    assert result.checks[0].status == "not_evaluable"


def test_no_prediction_is_distinct_from_success() -> None:
    before, after = frames()
    result = verify_outcome(None, grid(before), grid(after))
    assert (result.status, result.checked, result.passed) == (
        "no_prediction",
        False,
        None,
    )
    assert result.delta.changed_cells == 2


def test_verification_contract_rejects_passed_without_check() -> None:
    before, after = frames()
    delta = compare_observations(grid(before), grid(after))
    with pytest.raises(ValidationError):
        VerificationResult(status="unchecked", checked=False, passed=True, delta=delta)


def test_legacy_verification_payloads_map_to_statuses() -> None:
    assert verification_status({"checked": False, "passed": True}) == "no_prediction"
    assert verification_status({"checked": True, "passed": False}) == "mismatched"
    assert verification_status({"checked": True, "passed": True}) == "matched"
    assert verification_status({"status": "unchecked"}) == "unchecked"
