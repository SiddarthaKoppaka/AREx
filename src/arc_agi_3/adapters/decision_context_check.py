"""Validate a decision's hypothesis mutations against the current belief
state before it is accepted.

`BeliefStore` already rejects a stale or duplicate mutation - that guard
stays as the final invariant. Without this check, that rejection surfaced
only at `apply_cognitive_decision` time, after the decision was already
accepted, which killed the whole episode instead of being repaired the
same cheap way as a malformed tool call.
"""

from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision

from .generation_errors import BeliefContractError


def check_decision_context(decision: CognitiveDecision, context: AgentContext) -> None:
    current = context.hypothesis_versions
    proposed_ids = [item.hypothesis_id for item in decision.hypothesis_proposals]
    if len(proposed_ids) != len(set(proposed_ids)):
        raise BeliefContractError("hypothesis_proposals contains duplicate IDs")
    for proposal in decision.hypothesis_proposals:
        if proposal.hypothesis_id in current:
            raise BeliefContractError(
                f"hypothesis_proposals id {proposal.hypothesis_id!r} already "
                "exists; use hypothesis_updates to revise it"
            )
        if proposal.version != 1:
            raise BeliefContractError(
                f"hypothesis_proposals id {proposal.hypothesis_id!r} must start "
                "at version 1"
            )
        for superseded in proposal.supersedes:
            if superseded not in current:
                raise BeliefContractError(
                    f"hypothesis_proposals id {proposal.hypothesis_id!r} "
                    f"supersedes unknown hypothesis {superseded!r}"
                )
    updated_ids = {item.hypothesis_id for item in decision.hypothesis_updates}
    if updated_ids & set(proposed_ids):
        raise BeliefContractError(
            "the same hypothesis ID cannot appear in both hypothesis_proposals "
            "and hypothesis_updates"
        )
    for update in decision.hypothesis_updates:
        version = current.get(update.hypothesis_id)
        if version is None:
            raise BeliefContractError(
                f"hypothesis_updates id {update.hypothesis_id!r} does not exist"
            )
        if version != update.expected_version:
            raise BeliefContractError(
                f"hypothesis_updates id {update.hypothesis_id!r} expected_version "
                f"{update.expected_version} does not match current version {version}"
            )
