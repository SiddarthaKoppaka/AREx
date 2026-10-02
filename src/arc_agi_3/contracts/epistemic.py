"""Deterministic evidence cues shown to the model; none is a belief or policy."""

from typing import Literal

from pydantic import Field

from .base import Contract
from .cognition import Hypothesis
from .observation import Action
from .verification import VerificationStatus


class PredictionTest(Contract):
    verification_event_id: str
    step_id: int = Field(ge=0)
    prediction_id: str | None = None
    status: VerificationStatus
    mismatches: tuple[str, ...] = ()


class HypothesisLedgerEntry(Contract):
    """A hypothesis plus every verification of predictions that cited it."""

    hypothesis: Hypothesis
    tests: tuple[PredictionTest, ...] = ()
    omitted_tests: int = Field(default=0, ge=0)
    unacknowledged_contradictions: tuple[str, ...] = ()  # latest few, by event ID
    unacknowledged_total: int = Field(default=0, ge=0)


class ActionOutcomeSummary(Contract):
    transition_event_id: str
    step_id: int = Field(ge=0)
    from_current_state: bool
    changed_cells: int = Field(ge=0)
    translations: tuple[str, ...] = ()
    verification_status: VerificationStatus | None = None


class ActionEvidence(Contract):
    action: Action
    attempts: int = Field(ge=0)
    attempts_from_current_state: int = Field(ge=0)
    hypotheses_targeted: tuple[str, ...] = ()
    recent_outcomes: tuple[ActionOutcomeSummary, ...] = ()


RepeatKind = Literal[
    "same_state_same_action",
    "same_action_same_prediction",
    "same_action_same_hypotheses",
]


class PriorAttempt(Contract):
    action_event_id: str
    transition_event_id: str | None = None
    match: RepeatKind
    changed_cells: int | None = None
    verification_status: VerificationStatus | None = None


class RepeatCue(Contract):
    equivalent_prior: tuple[PriorAttempt, ...] = ()
    justification: str | None = None

    @property
    def unjustified(self) -> bool:
        return bool(self.equivalent_prior) and not self.justification


class ContextStats(Contract):
    recent_events_included: int = Field(ge=0)
    history_events: int = Field(ge=0)
    episodic_items_included: int = Field(default=0, ge=0)
    episodic_step_ids: tuple[int, ...] = ()
    episodic_query: str = ""
    hypotheses_included: int = Field(default=0, ge=0)
    omitted_hypothesis_ids: tuple[str, ...] = ()
