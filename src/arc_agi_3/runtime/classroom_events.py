"""Run one Student report or peer review and record it to the trace.

Split out of `classroom_adapter` so each half of the round stays small and
independently testable.
"""

from pydantic import JsonValue

from arc_agi_3.adapters.inference import InferenceBackend
from arc_agi_3.classroom_config import ClassroomConfig
from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.decision import AgentContext, ModelAttempt
from arc_agi_3.contracts.enums import EventType, StudentRole
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.peer_review import PeerReview

from .classroom_round import generate_peer_review, generate_student_report
from .io import EpisodeIO
from .usage import record_model_usage


def _attempt_payloads(attempts: tuple[ModelAttempt, ...]) -> list[JsonValue]:
    return [item.model_dump(mode="json", exclude_none=True) for item in attempts]


def run_student(
    io: EpisodeIO,
    backend: InferenceBackend,
    config: ClassroomConfig,
    role: StudentRole,
    context: AgentContext,
    history: list[EventEnvelope],
) -> StudentReport:
    student_id = f"S-{role.value}"
    report, usage, attempts = generate_student_report(
        backend,
        role,
        student_id,
        context,
        history,
        max_repairs=config.max_repairs,
        persist_invalid_output=config.persist_invalid_output,
        preview_chars=config.preview_chars,
        own_history_limit=config.own_history_limit,
    )
    cause = context.current_observation_event_id
    payload: dict[str, JsonValue] = {
        "report": report.model_dump(mode="json"),
        "attempts": _attempt_payloads(attempts),
    }
    event = io.append(
        EventType.STUDENT_REPORT,
        "classroom",
        context.turn,
        payload,
        (cause,) if cause else (),
    )
    record_model_usage(io, usage, context.turn, event.event_id)
    return report


def run_review(
    io: EpisodeIO,
    backend: InferenceBackend,
    config: ClassroomConfig,
    own_report: StudentReport,
    reports: tuple[StudentReport, ...],
) -> PeerReview:
    peers = tuple(item for item in reports if item.student_id != own_report.student_id)
    review, usage, attempts = generate_peer_review(
        backend,
        own_report,
        peers,
        max_repairs=config.max_repairs,
        persist_invalid_output=config.persist_invalid_output,
        preview_chars=config.preview_chars,
    )
    payload: dict[str, JsonValue] = {
        "review": review.model_dump(mode="json"),
        "attempts": _attempt_payloads(attempts),
    }
    event = io.append(EventType.PEER_REVIEW, "classroom", own_report.turn, payload)
    record_model_usage(io, usage, own_report.turn, event.event_id)
    return review
