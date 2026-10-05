"""Pure generation calls for one Student report or peer review.

No event recording here — `classroom_adapter.ClassroomModelAdapter` owns
that, so this stays easy to test without a trace store.
"""

from arc_agi_3.adapters.classroom_prompts import (
    peer_review_prompt,
    peer_review_schema,
    student_prompt,
    student_report_schema,
)
from arc_agi_3.adapters.inference import InferenceBackend
from arc_agi_3.adapters.structured_generation import run_structured
from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.decision import AgentContext, ModelAttempt, ModelUsage
from arc_agi_3.contracts.enums import StudentRole
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.peer_review import PeerReview
from arc_agi_3.evidence import recent_reports


def generate_student_report(
    backend: InferenceBackend,
    role: StudentRole,
    student_id: str,
    context: AgentContext,
    history: list[EventEnvelope],
    *,
    max_repairs: int,
    persist_invalid_output: bool,
    preview_chars: int,
    own_history_limit: int,
) -> tuple[StudentReport, ModelUsage, tuple[ModelAttempt, ...]]:
    own_history = (
        recent_reports(history, student_id, own_history_limit)
        if role is StudentRole.SKEPTIC
        else ()
    )
    prompt = student_prompt(role, student_id, context, own_history)
    parsed, usage, attempts, _ = run_structured(
        backend,
        student_report_schema(),
        StudentReport,
        lambda: (prompt, None),
        max_repairs,
        persist_invalid_output=persist_invalid_output,
        preview_chars=preview_chars,
        turn=context.turn,
    )
    return parsed, usage, attempts


def generate_peer_review(
    backend: InferenceBackend,
    own_report: StudentReport,
    peer_reports: tuple[StudentReport, ...],
    *,
    max_repairs: int,
    persist_invalid_output: bool,
    preview_chars: int,
) -> tuple[PeerReview, ModelUsage, tuple[ModelAttempt, ...]]:
    prompt = peer_review_prompt(
        own_report.student_id,
        own_report.role,
        own_report,
        peer_reports,
        own_report.turn,
    )
    parsed, usage, attempts, _ = run_structured(
        backend,
        peer_review_schema(),
        PeerReview,
        lambda: (prompt, None),
        max_repairs,
        persist_invalid_output=persist_invalid_output,
        preview_chars=preview_chars,
        turn=own_report.turn,
    )
    return parsed, usage, attempts
