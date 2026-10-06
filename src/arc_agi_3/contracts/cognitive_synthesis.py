"""Mechanical aggregation of specialist cognition for the core agent's
prompt.

Like the hypothesis ledger, this groups and flattens what specialists said.
It never decides whose claim is correct — the core agent reads this and
revises its own authoritative beliefs explicitly.
"""

from .base import Contract
from .cognitive_faculty import CognitiveFaculty
from .specialist_report import SpecialistReport
from .specialist_review import SpecialistReview


class SpecialistHypothesisRef(Contract):
    specialist_id: str
    faculty: CognitiveFaculty
    claim: str
    confidence: float


class CognitiveSynthesis(Contract):
    turn: int
    reports: tuple[SpecialistReport, ...]
    reviews: tuple[SpecialistReview, ...] = ()
    candidate_hypotheses: tuple[SpecialistHypothesisRef, ...] = ()
    flagged_contradictions: tuple[str, ...] = ()
