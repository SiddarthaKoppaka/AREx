"""Check whether a skill's precondition holds against the current
decision context - never against a hardcoded game solution."""

from arc_agi_3.contracts.decision import AgentContext

from .contracts import Skill


def check_precondition(skill: Skill, context: AgentContext) -> str | None:
    """None if applicable; otherwise the reason it is not."""
    ledger = {
        entry.hypothesis.hypothesis_id: entry.hypothesis
        for entry in context.hypothesis_ledger
    }
    for hypothesis_id in skill.precondition.required_hypothesis_ids:
        hypothesis = ledger.get(hypothesis_id)
        if hypothesis is None:
            return f"required hypothesis {hypothesis_id!r} is not in the ledger"
        if hypothesis.probability < skill.precondition.min_confidence:
            return f"hypothesis {hypothesis_id!r} is below the required confidence"
    facts = (
        context.working_scratchpad.verified_facts if context.working_scratchpad else ()
    )
    known_fact_ids = {fact.fact_id for fact in facts}
    missing_facts = set(skill.precondition.required_verified_fact_ids) - known_fact_ids
    if missing_facts:
        return f"required verified facts are missing: {sorted(missing_facts)}"
    return None


def is_eligible_for_game(skill: Skill, game_id: str) -> bool:
    """Per-game isolation: a skill is only usable on the game it was
    learned on unless it explicitly declares itself transfer-eligible."""
    return skill.game_id == game_id or skill.transfer_eligible
