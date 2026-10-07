"""Pure generation calls for one specialist report or review.

No event recording here — `runtime.specialist_tools` owns that, so this
stays easy to test without a trace store.
"""

from arc_agi_3.adapters.inference import InferenceBackend
from arc_agi_3.adapters.specialist_prompts import (
    specialist_prompt,
    specialist_report_schema,
    specialist_review_prompt,
    specialist_review_schema,
)
from arc_agi_3.adapters.structured_generation import run_structured
from arc_agi_3.contracts.decision import AgentContext, ModelAttempt, ModelUsage
from arc_agi_3.contracts.enums import CognitiveFaculty
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.contracts.specialist_review import SpecialistReview
from arc_agi_3.evidence import recent_reports


def generate_specialist_report(
    backend: InferenceBackend,
    faculty: CognitiveFaculty,
    specialist_id: str,
    context: AgentContext,
    history: list[EventEnvelope],
    *,
    max_repairs: int,
    persist_invalid_output: bool,
    preview_chars: int,
    own_history_limit: int,
) -> tuple[SpecialistReport, ModelUsage, tuple[ModelAttempt, ...]]:
    own_history = (
        recent_reports(history, specialist_id, own_history_limit)
        if faculty is CognitiveFaculty.CRITIC
        else ()
    )
    prompt = specialist_prompt(faculty, specialist_id, context, own_history)
    parsed, usage, attempts, _ = run_structured(
        backend,
        specialist_report_schema(),
        SpecialistReport,
        lambda: (prompt, None),
        max_repairs,
        persist_invalid_output=persist_invalid_output,
        preview_chars=preview_chars,
        turn=context.turn,
    )
    return parsed, usage, attempts


def generate_specialist_review(
    backend: InferenceBackend,
    own_report: SpecialistReport,
    other_reports: tuple[SpecialistReport, ...],
    *,
    max_repairs: int,
    persist_invalid_output: bool,
    preview_chars: int,
) -> tuple[SpecialistReview, ModelUsage, tuple[ModelAttempt, ...]]:
    prompt = specialist_review_prompt(
        own_report.specialist_id,
        own_report.faculty,
        own_report,
        other_reports,
        own_report.turn,
    )
    parsed, usage, attempts, _ = run_structured(
        backend,
        specialist_review_schema(),
        SpecialistReview,
        lambda: (prompt, None),
        max_repairs,
        persist_invalid_output=persist_invalid_output,
        preview_chars=preview_chars,
        turn=own_report.turn,
    )
    return parsed, usage, attempts
