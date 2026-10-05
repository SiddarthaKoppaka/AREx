"""Harness-computed aggregation of one classroom round.

Mechanical only, like the hypothesis ledger: this groups and
cross-references what Students and peer reviews said; it never decides
whose claim is correct. The Teacher reads this and decides.
"""

from pydantic import Field

from .base import Contract
from .classroom import StudentReport
from .enums import StudentRole
from .peer_review import PeerReview


class StudentHypothesisRef(Contract):
    student_id: str
    role: StudentRole
    claim: str
    confidence: float = Field(ge=0.0, le=1.0)


class ClassroomSynthesis(Contract):
    turn: int = Field(ge=0)
    reports: tuple[StudentReport, ...]
    peer_reviews: tuple[PeerReview, ...] = ()
    candidate_hypotheses: tuple[StudentHypothesisRef, ...] = ()
    flagged_contradictions: tuple[str, ...] = ()
