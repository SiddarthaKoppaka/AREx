"""Minimal public-workspace prompt projection."""

from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.trace.canonical import canonical_json


def decision_prompt(
    context: AgentContext,
    *,
    prior_output: str | None = None,
    validation_error: str | None = None,
) -> str:
    base = (
        "You are the agent. Return one CognitiveDecision as JSON matching the "
        "provided schema. Own all semantic interpretation, intent, strategy, and "
        "action choice. Do not provide private chain-of-thought; use only the "
        "public assessment and intent fields. Structural invariants: execute mode "
        "must set exactly one of action or action_chunk; every other mode must set "
        "both to null. Recover mode must set recovery; every other mode must set "
        "recovery to null.\nCONTEXT:\n" + canonical_json(context)
    )
    if validation_error is None:
        return base
    return (
        base
        + "\nYour previous output was structurally invalid. Preserve your decision "
        "semantics and return corrected JSON only.\nVALIDATION_ERROR:\n"
        + validation_error
        + "\nPREVIOUS_OUTPUT:\n"
        + (prior_output or "")
    )
