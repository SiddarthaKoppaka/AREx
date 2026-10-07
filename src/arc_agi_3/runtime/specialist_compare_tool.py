"""Tool handler for comparing two prior specialist reports. ADR-0009.

Split out of `specialist_tools.py` to keep each handler small; split from
`compare_specialist_reports_tool` itself for the same reason the plan's
`generate_specialist_review` stays a separate call from
`generate_specialist_report`.
"""

from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.enums import CognitiveFaculty, EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.contracts.specialist_tools import SpecialistCompareRequest
from arc_agi_3.evidence import recent_reports

from .io import EpisodeIO
from .specialist_round import generate_specialist_review
from .specialist_tools import attempt_payloads
from .usage import record_model_usage


def _latest_report(
    history: list[EventEnvelope], faculty: CognitiveFaculty
) -> SpecialistReport:
    found = recent_reports(history, f"S-{faculty.value}", 1)
    if not found:
        raise ValueError(f"no report yet for faculty: {faculty.value!r}")
    return found[0]


def compare_specialist_reports_tool(
    io: EpisodeIO, request: ToolRequest, requested: EventEnvelope, step: int
) -> ToolResult:
    if io.exocortex is None:
        raise ValueError("no exocortex backend configured for this run")
    args = SpecialistCompareRequest.model_validate(request.arguments)
    history = io.events.read()
    own = _latest_report(history, args.primary_faculty)
    others = tuple(_latest_report(history, faculty) for faculty in args.other_faculties)
    review, usage, attempts = generate_specialist_review(
        io.exocortex.backend,
        own,
        others,
        max_repairs=io.exocortex.config.max_repairs,
        persist_invalid_output=io.exocortex.config.persist_invalid_output,
        preview_chars=io.exocortex.config.preview_chars,
    )
    event = io.append(
        EventType.SPECIALIST_REVIEW,
        "exocortex",
        step,
        {
            "review": review.model_dump(mode="json"),
            "attempts": attempt_payloads(attempts),
        },
        (requested.event_id,),
    )
    record_model_usage(io, usage, step, event.event_id)
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"review": review.model_dump(mode="json")},
        evidence_refs=(event.event_id,),
    )
