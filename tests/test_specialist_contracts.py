"""SpecialistReport/SpecialistReview validate and round-trip correctly."""

from arc_agi_3.contracts.enums import CognitiveFaculty
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.prediction import Experiment
from arc_agi_3.contracts.specialist_report import (
    ActionProposal,
    Assumption,
    CandidateHypothesis,
    Contradiction,
    ExperimentProposal,
    SpecialistReport,
)
from arc_agi_3.contracts.specialist_review import SpecialistCritique, SpecialistReview


def test_specialist_report_round_trips_every_subtype() -> None:
    report = SpecialistReport(
        specialist_id="S-hypothesis",
        faculty=CognitiveFaculty.HYPOTHESIS,
        turn=3,
        assessment="Two candidate explanations remain.",
        evidence_refs=("event-1",),
        assumptions=(
            Assumption(claim="value 11 is static", evidence_refs=("event-1",)),
        ),
        hypotheses=(CandidateHypothesis(claim="11 is a goal", confidence=0.6),),
        contradictions=(
            Contradiction(description="11 changed twice", evidence_refs=("event-2",)),
        ),
        experiment_proposals=(
            ExperimentProposal(
                experiment=Experiment(experiment_id="e1", question="does 11 move?"),
                action=Action(action_id=1),
            ),
        ),
        action_proposals=(
            ActionProposal(action=Action(action_id=2), rationale="probe"),
        ),
        confidence=0.6,
        questions_for_peers=("is 11 state-dependent?",),
    )
    restored = SpecialistReport.model_validate(report.model_dump(mode="json"))
    assert restored == report


def test_specialist_review_round_trips() -> None:
    review = SpecialistReview(
        specialist_id="S-critic",
        faculty=CognitiveFaculty.CRITIC,
        turn=3,
        critiques=(
            SpecialistCritique(
                target_specialist_id="S-hypothesis",
                target_claim="11 is a goal",
                critique="evidence only shows cells changed, not a goal",
            ),
        ),
        agreements=("world model geometry is plausible",),
    )
    assert SpecialistReview.model_validate(review.model_dump(mode="json")) == review
