"""Persist LM-authored workspace mutations and bounded tool results."""

from arc_agi_3.ablations import require_capability
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope

from .io import EpisodeIO
from .task_validation import validate_task_updates
from .tools import run_tool


def apply_cognitive_decision(
    io: EpisodeIO,
    decision_event: EventEnvelope,
    decision: CognitiveDecision,
    step: int,
) -> None:
    known = {
        event.event_id
        for event in io.events.read()
        if event.event_id != decision_event.event_id
    }
    validate_task_updates(io, decision.task_updates, known)
    io.workspace.tasks.validate_many(decision.task_updates)
    prepared_scratchpad = None
    if decision.scratchpad_updates is not None:
        prepared_scratchpad = io.workspace.scratchpad.prepare(
            decision.scratchpad_updates, known
        )
    if decision.hypothesis_proposals or decision.hypothesis_updates:
        require_capability(io.config.ablations.hypotheses, "hypotheses")
    if decision.task_updates:
        require_capability(io.config.ablations.tasks, "tasks")
    if decision.world_model_updates:
        require_capability(io.config.ablations.world_models, "world_models")
    proposals = io.workspace.beliefs.add_many(decision.hypothesis_proposals)
    if proposals:
        io.append(
            EventType.HYPOTHESIS,
            "beliefs",
            step,
            {"versions": [item.model_dump(mode="json") for item in proposals]},
            (decision_event.event_id,),
        )
    for request in decision.hypothesis_updates:
        versions = io.workspace.beliefs.apply(request)
        io.append(
            EventType.BELIEF_UPDATE,
            "beliefs",
            step,
            {
                "request": request.model_dump(mode="json"),
                "versions": [item.model_dump(mode="json") for item in versions],
            },
            (decision_event.event_id,),
        )
    tasks = io.workspace.tasks.apply_many(decision.task_updates)
    if tasks:
        io.append(
            EventType.TASK_UPDATE,
            "tasks",
            step,
            {"versions": [item.model_dump(mode="json") for item in tasks]},
            (decision_event.event_id,),
        )
    models = io.workspace.world_models.add_many(decision.world_model_updates)
    if models:
        io.append(
            EventType.WORLD_MODEL,
            "world_model",
            step,
            {"versions": [item.model_dump(mode="json") for item in models]},
            (decision_event.event_id,),
        )
    if prepared_scratchpad is not None:
        io.workspace.scratchpad.commit(prepared_scratchpad)
        io.append(
            EventType.SCRATCHPAD_UPDATE,
            "working_memory",
            step,
            {"scratchpad": prepared_scratchpad.model_dump(mode="json")},
            (decision_event.event_id,),
        )
    for tool_request in decision.tool_requests:
        run_tool(io, tool_request, decision_event, step)
