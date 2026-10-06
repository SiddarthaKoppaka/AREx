"""A specialist's advisory critique of another faculty's report — equally
non-authoritative."""

from pydantic import Field

from .base import Contract
from .cognitive_faculty import CognitiveFaculty


class SpecialistCritique(Contract):
    target_specialist_id: str = Field(min_length=1)
    target_claim: str = Field(min_length=1, max_length=240)
    critique: str = Field(min_length=1, max_length=240)
    evidence_refs: tuple[str, ...] = ()


class SpecialistReview(Contract):
    specialist_id: str = Field(min_length=1)
    faculty: CognitiveFaculty
    turn: int = Field(ge=0)
    critiques: tuple[SpecialistCritique, ...] = Field(default=(), max_length=8)
    agreements: tuple[str, ...] = Field(default=(), max_length=6)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=6)
