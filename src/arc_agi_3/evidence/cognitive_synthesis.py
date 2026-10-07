"""Mechanical aggregation of every specialist report/review so far this
episode, for the core agent's context.

Like the hypothesis ledger, this groups and flattens what specialists
said. It never decides whose claim is correct — the core agent reads this
and revises its own authoritative beliefs explicitly. Rebuilt fresh from
the immutable trace each turn, the same as every other context view, so a
report from turn 1 is still visible in turn 5's synthesis with no
re-consultation.
"""

from arc_agi_3.contracts.cognitive_synthesis import (
    CognitiveSynthesis,
    SpecialistHypothesisRef,
)
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.contracts.specialist_review import SpecialistReview


def build_cognitive_synthesis(
    history: list[EventEnvelope],
) -> CognitiveSynthesis | None:
    reports = tuple(
        SpecialistReport.model_validate(event.payload["report"])
        for event in history
        if event.event_type is EventType.SPECIALIST_REPORT
    )
    if not reports:
        return None
    reviews = tuple(
        SpecialistReview.model_validate(event.payload["review"])
        for event in history
        if event.event_type is EventType.SPECIALIST_REVIEW
    )
    candidate_hypotheses = tuple(
        SpecialistHypothesisRef(
            specialist_id=report.specialist_id,
            faculty=report.faculty,
            claim=hypothesis.claim,
            confidence=hypothesis.confidence,
        )
        for report in reports
        for hypothesis in report.hypotheses
    )
    flagged = tuple(
        f"{review.specialist_id} on {critique.target_specialist_id}: "
        f"{critique.critique}"
        for review in reviews
        for critique in review.critiques
    )
    return CognitiveSynthesis(
        turn=max(report.turn for report in reports),
        reports=reports,
        reviews=reviews,
        candidate_hypotheses=candidate_hypotheses,
        flagged_contradictions=flagged,
    )
