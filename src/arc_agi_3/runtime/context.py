"""Bounded working set: current evidence, live hypotheses, and retrievable refs."""

from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig
from arc_agi_3.context import compact_context_event, context_event, summarize_episode
from arc_agi_3.context.projection_keys import HIDDEN_IN_CONTEXT
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import TaskStatus
from arc_agi_3.contracts.epistemic import ContextStats
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.recovery import reconstruct_recovery
from arc_agi_3.trace.store import JsonlEventStore

from .recent_events import event_view, recent_turn_events
from .working_set import epistemic_fields


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
    visible = [e for e in history if e.event_type not in HIDDEN_IN_CONTEXT]
    recent = recent_turn_events(
        visible if context_compaction else history, raw_recent_turns, recent_event_limit
    )
    recent_ids = {event.event_id for event in recent}
    older = [event for event in history if event.event_id not in recent_ids]
    project = compact_context_event if context_compaction else context_event
    task_views = workspace.tasks.views() if ablations.tasks else ()
    epistemic = epistemic_fields(history, observation, workspace, ablations)
    omitted = epistemic.pop("omitted_hypothesis_ids", ())
    query = " ".join(
        (
            workspace.scratchpad.current.objective,
            *(v.task.purpose for v in task_views if v.task.status is TaskStatus.OPEN),
            *(item.claim for item in epistemic.get("hypotheses", ())),
        )
    )
    memory = (
        summarize_episode(older, episodic_retrieval_limit, query)
        if ablations.persistent_memory
        else None
    )
    recovery = reconstruct_recovery(
        history if ablations.persistent_memory else [], compact=context_compaction
    )
    # The divergence window is shown while it is recent; it stays retrievable.
    recent_window = not context_compaction or (
        recovery is not None and recovery.divergence_event_id in recent_ids
    )
    stats = ContextStats(
        recent_events_included=len(recent),
        history_events=len(history),
        episodic_items_included=len(memory.items) if memory else 0,
        episodic_step_ids=tuple(item.step_id for item in memory.items)
        if memory
        else (),
        episodic_query=query[:240],
        hypotheses_included=len(epistemic.get("hypotheses", ())),
        omitted_hypothesis_ids=omitted,
    )
    return AgentContext(
        turn=turn,
        observation=observation,
        budget=ledger.remaining(),
        recent_event_refs=tuple(event.event_id for event in recent),
        recent_events=tuple(
            event_view(events, event, project, context_compaction) for event in recent
        ),
        working_scratchpad=workspace.scratchpad.current,
        episodic_memory=memory,
        tasks=task_views,
        world_models=workspace.world_models.current if ablations.world_models else (),
        recovery_evidence=recovery if recovery and recent_window else None,
        context_stats=stats,
        **epistemic,
    )
