"""Create trace-linked runtime checkpoints."""

from pydantic import JsonValue

from arc_agi_3.adapters.protocols import EnvironmentAdapter
from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.events import CheckpointRecord
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.trace.canonical import canonical_hash
from arc_agi_3.trace.checkpoints import CheckpointStore
from arc_agi_3.trace.store import JsonlEventStore


def save_checkpoint(
    config: RunConfig,
    step_id: int,
    observation: Observation,
    environment: EnvironmentAdapter,
    ledger: BudgetLedger,
    events: JsonlEventStore,
    checkpoints: CheckpointStore,
    cognitive_state: dict[str, JsonValue] | None = None,
    checkpoint_id: str | None = None,
) -> CheckpointRecord:
    if events.last_hash is None:
        raise ValueError("cannot checkpoint before the first event")
    draft = CheckpointRecord(
        checkpoint_id=checkpoint_id or f"checkpoint-{step_id:06d}",
        run_id=config.run_id,
        episode_id=config.episode_id,
        branch_id=events.branch_id,
        step_id=step_id,
        last_event_hash=events.last_hash,
        observation=observation.model_dump(mode="json"),
        budget=ledger.snapshot().model_dump(mode="json"),
        environment_state=environment.checkpoint(),
        cognitive_state=cognitive_state or {},
        checksum="",
    )
    digest = canonical_hash(draft.model_dump(exclude={"checksum"}))
    checkpoint = draft.model_copy(update={"checksum": digest})
    checkpoints.save(checkpoint)
    return checkpoint
