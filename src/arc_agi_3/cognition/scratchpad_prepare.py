"""Single source of truth for what a ScratchpadUpdates produces.

Both the real commit path (`ScratchpadStore.prepare`) and the early
contract check (`adapters/scratchpad_contract_check.py`) call this, so the
two can never drift apart.
"""

from arc_agi_3.contracts.scratchpad import ScratchpadUpdates, WorkingScratchpad
from arc_agi_3.trace.canonical import canonical_json

from .scratchpad_merge import merge_facts, merge_next_test


def prepare_scratchpad(
    before: WorkingScratchpad, updates: ScratchpadUpdates, valid_evidence: set[str]
) -> tuple[WorkingScratchpad, int]:
    """Merge updates into `before` and estimate the resulting token cost.

    Raises ValueError on any contract violation (bad evidence refs,
    self-supersede, duplicate/missing supersedes, dropped evidence, wrong
    version) - never checks the token budget itself, since the two callers
    compare the estimate against different budget sources.
    """
    next_test, revisions = merge_next_test(before, updates)
    questions = [
        item
        for item in before.open_questions
        if item not in updates.resolve_open_questions
    ]
    questions.extend(
        item for item in updates.add_open_questions if item not in questions
    )
    values = {
        "version": before.version + 1,
        "objective": updates.set_objective or before.objective,
        "verified_facts": merge_facts(before, updates, valid_evidence),
        "action_model": {**before.action_model, **updates.action_model_updates},
        "active_plan": updates.revise_plan
        if updates.revise_plan is not None
        else before.active_plan,
        "open_questions": tuple(questions),
        "last_useful_result": updates.set_last_useful_result
        or before.last_useful_result,
        "next_test": next_test,
        "plan_revisions": revisions,
        "capabilities": before.capabilities,
    }
    updated = before.model_copy(update=values)
    estimated_tokens = (len(canonical_json(updated)) + 3) // 4
    return updated, estimated_tokens
