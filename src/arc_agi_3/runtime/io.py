"""Trace, budget, context, and checkpoint services for the episode loop."""

from pydantic import JsonValue

from arc_agi_3.adapters.protocols import EnvironmentAdapter
from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.recovery import BranchTracker
from arc_agi_3.trace.checkpoints import CheckpointStore
from arc_agi_3.trace.store import JsonlEventStore

from .checkpointing import save_checkpoint
from .context import project_context


class EpisodeIO:
    def __init__(
        self,
        config: RunConfig,
        environment: EnvironmentAdapter,
        events: JsonlEventStore,
        checkpoints: CheckpointStore,
        ledger: BudgetLedger,
    ) -> None:
        self.config, self.environment = config, environment
        self.events, self.checkpoints, self.ledger = events, checkpoints, ledger
        self.workspace = CognitiveWorkspace(config.belief)
        self.branches = BranchTracker(config.branch_id)

    def append(
        self,
        kind: EventType,
        component: str,
        step: int,
        payload: dict[str, JsonValue],
        causes: tuple[str, ...] = (),
    ) -> EventEnvelope:
        budget = self.ledger.snapshot().model_dump(mode="json")
        return self.events.append(
            kind, component, step, payload, causal_refs=causes, budget=budget
        )

    def spend(
        self,
        resource: Resource,
        amount: int,
        step: int,
        causes: tuple[str, ...] = (),
    ) -> EventEnvelope:
        self.ledger.spend(resource, amount)
        payload: dict[str, JsonValue] = {"resource": resource, "amount": amount}
        return self.append(EventType.BUDGET, "budget", step, payload, causes)

    def context(self, turn: int, observation: Observation) -> AgentContext:
        return project_context(
            turn,
            observation,
            self.ledger,
            self.events,
            self.workspace,
            self.config.ablations,
            self.config.recent_event_limit,
            self.config.context_compaction,
        )

    def checkpoint(
        self,
        step: int,
        observation: Observation,
        cause: str,
        suffix: int | None = None,
    ) -> None:
        checkpoint_id = (
            f"checkpoint-{step:06d}-{suffix:03d}" if suffix is not None else None
        )
        checkpoint = save_checkpoint(
            self.config,
            step,
            observation,
            self.environment,
            self.ledger,
            self.events,
            self.checkpoints,
            self.workspace.snapshot(),
            checkpoint_id,
        )
        payload: dict[str, JsonValue] = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "checksum": checkpoint.checksum,
        }
        self.append(EventType.CHECKPOINT, "checkpoint", step, payload, (cause,))
