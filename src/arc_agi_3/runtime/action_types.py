"""State passed across the authorize/observe environment boundary."""

from dataclasses import dataclass

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ExpectedOutcome
from arc_agi_3.contracts.observation import Action, Observation
from arc_agi_3.contracts.verification import VerificationResult


@dataclass(frozen=True)
class AuthorizedAction:
    turn: int
    before: Observation
    decision: EventEnvelope
    action: Action
    expected: ExpectedOutcome | None
    action_event: EventEnvelope
    checkpoint: bool
    checkpoint_suffix: int | None = None


@dataclass(frozen=True)
class ActionOutcome:
    observation: Observation
    observed_event: EventEnvelope
    verification: VerificationResult
    verification_event: EventEnvelope
