"""Tool handler for core-agent-requested specialist consultation. ADR-0009.

Unlike every other tool in `tool_handlers.py`, this triggers a real backend
generation through the same bounded engine specialists always used — it is
the one place the tool-execution path needs the current observation.
"""

from pydantic import JsonValue

from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.model_io import ModelAttempt
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.specialist_tools import SpecialistConsultRequest

from .io import EpisodeIO
from .specialist_round import generate_specialist_report
from .usage import record_model_usage


def attempt_payloads(attempts: tuple[ModelAttempt, ...]) -> list[JsonValue]:
    return [item.model_dump(mode="json", exclude_none=True) for item in attempts]


def consult_specialist_tool(
    io: EpisodeIO,
    request: ToolRequest,
    requested: EventEnvelope,
    observation: Observation,
    step: int,
) -> ToolResult:
    if io.exocortex is None:
        raise ValueError("no exocortex backend configured for this run")
    args = SpecialistConsultRequest.model_validate(request.arguments)
    if args.faculty not in io.exocortex.config.faculties:
        raise ValueError(f"faculty not available: {args.faculty.value!r}")
    specialist_id = f"S-{args.faculty.value}"
    context = io.context(step, observation)
    report, usage, attempts = generate_specialist_report(
        io.exocortex.backend,
        args.faculty,
        specialist_id,
        context,
        io.events.read(),
        max_repairs=io.exocortex.config.max_repairs,
        persist_invalid_output=io.exocortex.config.persist_invalid_output,
        preview_chars=io.exocortex.config.preview_chars,
        own_history_limit=io.exocortex.config.own_history_limit,
    )
    cause = context.current_observation_event_id
    causes = (requested.event_id, cause) if cause else (requested.event_id,)
    event = io.append(
        EventType.SPECIALIST_REPORT,
        "exocortex",
        step,
        {
            "report": report.model_dump(mode="json"),
            "attempts": attempt_payloads(attempts),
        },
        causes,
    )
    record_model_usage(io, usage, step, event.event_id)
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"report": report.model_dump(mode="json")},
        evidence_refs=(event.event_id,),
    )
