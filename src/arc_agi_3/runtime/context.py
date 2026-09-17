"""Lossless recent window plus current deterministic workspace projection."""

from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig
from arc_agi_3.context import context_event
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
) -> AgentContext:
    recent = events.read()[-8:]
    memory = events.read() if ablations.persistent_memory else []
    context = AgentContext(
        turn=turn,
        observation=observation,
        budget=ledger.remaining(),
        recent_event_refs=tuple(event.event_id for event in recent),
        recent_events=tuple(context_event(event) for event in recent),
        hypotheses=workspace.beliefs.current if ablations.hypotheses else (),
        tasks=workspace.tasks.views() if ablations.tasks else (),
        world_models=workspace.world_models.current if ablations.world_models else (),
    )
    return context.model_copy(
        update={"recovery_evidence": reconstruct_recovery(memory)}
    )
