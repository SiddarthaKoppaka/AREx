"""StudentReport/PeerReview/ClassroomSynthesis validate and aggregate correctly."""

from arc_agi_3.contracts.classroom import (
    ActionProposal,
    Assumption,
    CandidateHypothesis,
    Contradiction,
    ExperimentProposal,
    StudentReport,
)
from arc_agi_3.contracts.enums import StudentRole
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.peer_review import PeerCritique, PeerReview
from arc_agi_3.contracts.prediction import Experiment
from arc_agi_3.evidence import build_synthesis


def test_student_report_round_trips_every_subtype() -> None:
    report = StudentReport(
        student_id="S-scientist",
        role=StudentRole.SCIENTIST,
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
    restored = StudentReport.model_validate(report.model_dump(mode="json"))
    assert restored == report


def test_peer_review_round_trips() -> None:
    review = PeerReview(
        student_id="S-skeptic",
        role=StudentRole.SKEPTIC,
        turn=3,
        critiques=(
            PeerCritique(
                target_student_id="S-scientist",
                target_claim="11 is a goal",
                critique="evidence only shows cells changed, not a goal",
            ),
        ),
        agreements=("world model geometry is plausible",),
    )
    assert PeerReview.model_validate(review.model_dump(mode="json")) == review


def test_build_synthesis_flattens_mechanically_without_interpreting() -> None:
    scientist = StudentReport(
        student_id="S-scientist",
        role=StudentRole.SCIENTIST,
        turn=1,
        assessment="x",
        hypotheses=(CandidateHypothesis(claim="11 is a goal", confidence=0.7),),
    )
    skeptic_review = PeerReview(
        student_id="S-skeptic",
        role=StudentRole.SKEPTIC,
        turn=1,
        critiques=(
            PeerCritique(
                target_student_id="S-scientist",
                target_claim="11 is a goal",
                critique="unsupported",
            ),
        ),
    )
    synthesis = build_synthesis(1, (scientist,), (skeptic_review,))
    assert synthesis.candidate_hypotheses[0].claim == "11 is a goal"
    assert synthesis.candidate_hypotheses[0].student_id == "S-scientist"
    assert synthesis.flagged_contradictions == (
        "S-skeptic on S-scientist: unsupported",
    )
