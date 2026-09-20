"""Current observation and bounded deterministic workspace projection."""

from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig
from arc_agi_3.context import compact_context_event, context_event
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.observation import Observation
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
) -> AgentContext:
    history = events.read()
    recent = history[-recent_event_limit:]
    memory = history if ablations.persistent_memory else []
    project = compact_context_event if context_compaction else context_event
    context = AgentContext(
        turn=turn,
        observation=observation,
        budget=ledger.remaining(),
        recent_event_refs=tuple(event.event_id for event in recent),
        recent_events=tuple(project(event) for event in recent),
        hypotheses=workspace.beliefs.current if ablations.hypotheses else (),
        tasks=workspace.tasks.views() if ablations.tasks else (),
        world_models=workspace.world_models.current if ablations.world_models else (),
    )
    evidence = reconstruct_recovery(memory, compact=context_compaction)
    return context.model_copy(update={"recovery_evidence": evidence})
