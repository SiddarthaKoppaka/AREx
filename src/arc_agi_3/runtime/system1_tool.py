"""Tool handler for Core-Agent-requested System-1 (JEV) consultation.

See ADR-0010. Suggest mode only: System-1 can score/select among
candidates the Core Agent itself constructed, but nothing here executes
anything - the Core Agent reads the result on a later turn and decides
what to do with it. execute_one delegation is deliberately not supported
yet (see ADR-0010's Consequences).
"""

from arc_agi_3.contracts.cognition import ToolRequest, ToolResult
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.system1 import System1DecisionRequest, System1DecisionResult

from .io import EpisodeIO


def _validate_candidates(
    args: System1DecisionRequest, result: System1DecisionResult
) -> None:
    candidate_ids = {item.candidate_id for item in args.candidates}
    if result.selected_candidate_id not in candidate_ids:
        raise ValueError(
            "System-1 selected an unauthorized candidate: "
            f"{result.selected_candidate_id!r}"
        )
    unknown = [
        item.candidate_id
        for item in result.candidate_probabilities
        if item.candidate_id not in candidate_ids
    ]
    if unknown:
        raise ValueError(f"System-1 scored unauthorized candidates: {unknown}")


def consult_system1_tool(
    io: EpisodeIO, request: ToolRequest, requested: EventEnvelope, step: int
) -> ToolResult:
    if io.system1 is None:
        raise ValueError("no System-1 backend configured for this run")
    args = System1DecisionRequest.model_validate(request.arguments)
    if args.execution_mode != "suggest":
        raise ValueError("execute_one delegation is not yet supported")
    result = io.system1.decide(args)
    _validate_candidates(args, result)
    event = io.append(
        EventType.SYSTEM1_DECISION,
        "exocortex",
        step,
        {
            "request": args.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        },
        (requested.event_id,),
    )
    return ToolResult(
        request_id=request.request_id,
        status="complete",
        output={"system1": result.model_dump(mode="json")},
        evidence_refs=(event.event_id,),
    )
