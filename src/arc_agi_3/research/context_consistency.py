"""Optional offline comparison of action choice under two context projections."""

from collections.abc import Callable

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.trace.canonical import canonical_hash


class NextActionConsistency(Contract):
    full_context_hash: str
    compact_context_hash: str
    same_mode: bool
    same_action_authorization: bool
    comparison_calls: int = Field(default=2, ge=2, le=2)


def compare_next_action(
    decide: Callable[[AgentContext], CognitiveDecision],
    full_context: AgentContext,
    compact_context: AgentContext,
) -> NextActionConsistency:
    """Run only in an offline diagnostic; this makes two model decisions."""
    if full_context.observation != compact_context.observation:
        raise ValueError("contexts must share the current observation")
    full = decide(full_context)
    compact = decide(compact_context)
    return NextActionConsistency(
        full_context_hash=canonical_hash(full_context),
        compact_context_hash=canonical_hash(compact_context),
        same_mode=full.mode == compact.mode,
        same_action_authorization=_authorization(full) == _authorization(compact),
    )


def _authorization(decision: CognitiveDecision) -> object:
    if decision.action is not None:
        return decision.action.model_dump(mode="json")
    if decision.action_chunk is not None:
        return decision.action_chunk.model_dump(mode="json")
    return None
