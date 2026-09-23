"""Current observation and bounded deterministic workspace projection."""

from collections.abc import Callable

from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig
from arc_agi_3.context import compact_context_event, context_event, summarize_episode
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import TaskStatus
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.retrieval import ContextEvent
from arc_agi_3.recovery import reconstruct_recovery
from arc_agi_3.trace.store import JsonlEventStore


def project_context(
    turn: int,
    observation: Observation,
    ledger: BudgetLedger,
    events: JsonlEventStore,
    workspace: CognitiveWorkspace,
    ablations: AblationConfig,
    recent_event_limit: int = 8,
    context_compaction: bool = True,
    raw_recent_turns: int = 2,
    episodic_retrieval_limit: int = 6,
) -> AgentContext:
    history = events.read()
    recent = _recent_turn_events(history, raw_recent_turns, recent_event_limit)
    recent_ids = {event.event_id for event in recent}
    older = [event for event in history if event.event_id not in recent_ids]
    memory = history if ablations.persistent_memory else []
    project = compact_context_event if context_compaction else context_event
    task_views = workspace.tasks.views() if ablations.tasks else ()
    active_tasks = " ".join(
        view.task.purpose
        for view in task_views
        if view.task.status is TaskStatus.OPEN and not view.structurally_blocked
    )
    context = AgentContext(
        turn=turn,
        observation=observation,
        budget=ledger.remaining(),
        recent_event_refs=tuple(event.event_id for event in recent),
        recent_events=tuple(
            _event_view(events, event, project, context_compaction) for event in recent
        ),
        working_scratchpad=workspace.scratchpad.current,
        episodic_memory=summarize_episode(
            older,
            episodic_retrieval_limit,
            f"{workspace.scratchpad.current.objective} {active_tasks}",
        )
        if ablations.persistent_memory
        else None,
        hypotheses=workspace.beliefs.current if ablations.hypotheses else (),
        tasks=task_views,
        world_models=workspace.world_models.current if ablations.world_models else (),
    )
    evidence = reconstruct_recovery(memory, compact=context_compaction)
    return context.model_copy(update={"recovery_evidence": evidence})


def _recent_turn_events(
    history: list[EventEnvelope], turn_limit: int, event_limit: int
) -> list[EventEnvelope]:
    if turn_limit <= 0 or not history:
        return []
    steps = sorted({event.step_id for event in history})[-turn_limit:]
    while len(steps) > 1:
        count = sum(event.step_id in steps for event in history)
        if count <= event_limit:
            break
        steps.pop(0)
    selected = set(steps)
    return [event for event in history if event.step_id in selected]


def _event_view(
    events: JsonlEventStore,
    event: EventEnvelope,
    project: Callable[[EventEnvelope], ContextEvent],
    compact: bool,
) -> ContextEvent:
    visible = project(event)
    reference = events.artifacts.reference(event) if compact else None
    if reference is None:
        return visible
    return visible.model_copy(
        update={"payload": visible.payload | {"artifact_ref": reference}}
    )
