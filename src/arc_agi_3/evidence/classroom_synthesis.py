"""Mechanical aggregation of one classroom round for the Teacher's prompt.

Like the hypothesis ledger, this groups and flattens what Students and
peer reviews said. It never decides whose claim is correct — the Teacher
reads this and revises its own authoritative beliefs explicitly.
"""

from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.classroom_synthesis import (
    ClassroomSynthesis,
    StudentHypothesisRef,
)
from arc_agi_3.contracts.peer_review import PeerReview


def build_synthesis(
    turn: int, reports: tuple[StudentReport, ...], peer_reviews: tuple[PeerReview, ...]
) -> ClassroomSynthesis:
    candidate_hypotheses = tuple(
        StudentHypothesisRef(
            student_id=report.student_id,
            role=report.role,
            claim=hypothesis.claim,
            confidence=hypothesis.confidence,
        )
        for report in reports
        for hypothesis in report.hypotheses
    )
    flagged = tuple(
        f"{review.student_id} on {critique.target_student_id}: {critique.critique}"
        for review in peer_reviews
        for critique in review.critiques
    )
    return ClassroomSynthesis(
        turn=turn,
        reports=reports,
        peer_reviews=peer_reviews,
        candidate_hypotheses=candidate_hypotheses,
        flagged_contradictions=flagged,
    )
