"""Validate scratchpad_updates against the current scratchpad before
acceptance.

Mirrors `check_decision_context`'s role for hypothesis mutations: without
this, a scratchpad contract violation surfaced only inside
`apply_cognitive_decision`, after the decision was already accepted,
ending the episode instead of being repaired the same cheap way as a
malformed tool call. `ScratchpadStore.prepare` still re-validates at
commit time as the final invariant.
"""

from arc_agi_3.cognition.scratchpad_prepare import prepare_scratchpad
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision

from .generation_errors import ScratchpadContractError


def check_scratchpad_update(decision: CognitiveDecision, context: AgentContext) -> None:
    updates = decision.scratchpad_updates
    if updates is None or context.working_scratchpad is None:
        return
    try:
        _, estimated_tokens = prepare_scratchpad(
            context.working_scratchpad, updates, set(context.known_event_ids)
        )
    except ValueError as error:
        raise ScratchpadContractError(str(error)) from error
    if estimated_tokens > context.scratchpad_token_budget:
        raise ScratchpadContractError(
            f"working scratchpad would exceed {context.scratchpad_token_budget} "
            "token budget; consolidate existing facts via supersedes"
        )
