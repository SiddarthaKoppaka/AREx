"""System1DecisionRequest/Result validate exactly what ADR-0010 requires."""

import pytest
from pydantic import ValidationError

from arc_agi_3.contracts.system1 import (
    CandidateOperation,
    CandidateProbability,
    System1DecisionRequest,
    System1DecisionResult,
)


def _candidate(candidate_id: str) -> CandidateOperation:
    return CandidateOperation(candidate_id=candidate_id, description="x")


def test_valid_candidate_set_round_trips() -> None:
    request = System1DecisionRequest(
        trial_id="t1",
        goal="goal",
        state_summary="state",
        candidates=(_candidate("a"), _candidate("b")),
    )
    assert System1DecisionRequest.model_validate(request.model_dump(mode="json")) == (
        request
    )


def test_duplicate_candidate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        System1DecisionRequest(
            trial_id="t1",
            goal="goal",
            state_summary="state",
            candidates=(_candidate("a"), _candidate("a")),
        )


def test_empty_candidate_set_is_rejected() -> None:
    with pytest.raises(ValidationError):
        System1DecisionRequest(
            trial_id="t1", goal="goal", state_summary="state", candidates=()
        )


def test_probability_out_of_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateProbability(candidate_id="a", probability=1.5)


def test_result_does_not_require_a_candidate_to_exist_on_its_own() -> None:
    # The contract itself has no cross-object knowledge of the request's
    # candidate set - that authorization check is the tool handler's job
    # (test_system1_tool.py), not something baked into this type.
    result = System1DecisionResult(trial_id="t1", selected_candidate_id="anything")
    assert result.selected_candidate_id == "anything"
