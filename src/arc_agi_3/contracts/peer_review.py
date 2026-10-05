"""A Student's advisory critique of its peers — equally non-authoritative."""

from pydantic import Field

from .base import Contract
from .enums import StudentRole


class PeerCritique(Contract):
    target_student_id: str = Field(min_length=1)
    target_claim: str = Field(min_length=1, max_length=240)
    critique: str = Field(min_length=1, max_length=240)
    evidence_refs: tuple[str, ...] = ()


class PeerReview(Contract):
    student_id: str = Field(min_length=1)
    role: StudentRole
    turn: int = Field(ge=0)
    critiques: tuple[PeerCritique, ...] = Field(default=(), max_length=8)
    agreements: tuple[str, ...] = Field(default=(), max_length=6)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=6)
