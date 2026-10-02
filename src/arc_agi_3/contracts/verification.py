"""Deterministic transition and prediction-verification results.

`status` separates four cases that must never be conflated:
matched / mismatched (claims were checked), unchecked (a prediction existed
but none of its claims was evaluable), and no_prediction. `passed` is only set
when something was checked. A mismatch is evidence for the model, never a
belief update.
"""

from typing import Literal

from pydantic import JsonValue, model_validator

from .base import Contract
from .transition import TransitionEvidence

StateDelta = TransitionEvidence
VerificationStatus = Literal["matched", "mismatched", "unchecked", "no_prediction"]
ClaimStatus = Literal["matched", "mismatched", "not_evaluable"]


class ClaimCheck(Contract):
    claim: str
    status: ClaimStatus
    expected: JsonValue = None
    observed: JsonValue = None


class VerificationResult(Contract):
    status: VerificationStatus
    checked: bool
    passed: bool | None = None
    mismatches: tuple[str, ...] = ()
    checks: tuple[ClaimCheck, ...] = ()
    prediction_id: str | None = None
    hypothesis_ids: tuple[str, ...] = ()
    prediction_signature: str | None = None
    delta: TransitionEvidence

    @model_validator(mode="after")
    def consistent(self) -> "VerificationResult":
        checked = self.status in {"matched", "mismatched"}
        if self.checked != checked:
            raise ValueError("checked must be true only for matched or mismatched")
        if self.passed != (None if not checked else self.status == "matched"):
            raise ValueError("passed must be unset unless a claim was checked")
        return self
