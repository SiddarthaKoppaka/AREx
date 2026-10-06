"""Advisory, non-authoritative cognition from a specialist faculty.

A SpecialistReport never crosses the agency boundary: it cannot execute an
action, mutate BeliefStore/TaskGraph/ScratchpadStore, or authorize a
side-effecting tool. Only the core agent's CognitiveDecision does that —
this is evidence the core agent may consult, nothing more. See ADR-0009.
"""

from pydantic import Field

from arc_agi_3.world_model.contracts import WorldModel

from .base import Contract
from .cognitive_faculty import CognitiveFaculty
from .observation import Action
from .prediction import ExpectedOutcome, Experiment


class Assumption(Contract):
    claim: str = Field(min_length=1, max_length=240)
    evidence_refs: tuple[str, ...] = ()


class CandidateHypothesis(Contract):
    """Pre-adoption, unversioned — the authoritative form is `Hypothesis`."""

    claim: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...] = ()
    contradicting_refs: tuple[str, ...] = ()


class Contradiction(Contract):
    description: str = Field(min_length=1, max_length=240)
    hypothesis_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class ExperimentProposal(Contract):
    experiment: Experiment
    action: Action
    expected_outcome: ExpectedOutcome | None = None


class ActionProposal(Contract):
    action: Action
    rationale: str = Field(min_length=1, max_length=240)
    expected_outcome: ExpectedOutcome | None = None


class SpecialistReport(Contract):
    specialist_id: str = Field(min_length=1)
    faculty: CognitiveFaculty
    turn: int = Field(ge=0)
    assessment: str = Field(min_length=1, max_length=512)
    evidence_refs: tuple[str, ...] = ()
    assumptions: tuple[Assumption, ...] = Field(default=(), max_length=8)
    hypotheses: tuple[CandidateHypothesis, ...] = Field(default=(), max_length=8)
    predictions: tuple[ExpectedOutcome, ...] = Field(default=(), max_length=4)
    contradictions: tuple[Contradiction, ...] = Field(default=(), max_length=8)
    experiment_proposals: tuple[ExperimentProposal, ...] = Field(
        default=(), max_length=4
    )
    action_proposals: tuple[ActionProposal, ...] = Field(default=(), max_length=4)
    world_model_fragment: WorldModel | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainties: tuple[str, ...] = Field(default=(), max_length=6)
    questions_for_peers: tuple[str, ...] = Field(default=(), max_length=6)
