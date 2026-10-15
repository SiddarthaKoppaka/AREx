"""Typed replay cases and per-model test reports. See handoff.md section
5.2. Every case links to an authoritative observed transition - nothing
here is a synthetic/predicted observation."""

from typing import Literal

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.observation import Action, Observation

Split = Literal["train", "holdout"]
CaseStatus = Literal["match", "mismatch", "error"]


class TransitionCase(Contract):
    case_id: str
    before_observation_event_id: str
    action_event_id: str
    after_transition_event_id: str
    before: Observation
    action: Action
    after: Observation
    split: Split


class CaseResult(Contract):
    case_id: str
    split: Split
    status: CaseStatus
    error_category: str | None = None


class ModelTestReport(Contract):
    model_id: str
    model_version: int
    total_cases: int = Field(ge=0)
    train_cases: int = Field(ge=0)
    holdout_cases: int = Field(ge=0)
    train_mismatches: int = Field(ge=0)
    holdout_mismatches: int = Field(ge=0)
    errors: int = Field(ge=0)
    train_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    holdout_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    counterexample_case_ids: tuple[str, ...] = ()
