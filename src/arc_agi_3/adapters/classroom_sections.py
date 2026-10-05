"""Per-role bounded evidence sections, built from the one shared AgentContext."""

from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.decision import AgentContext

from .observation_projection import observation_view
from .prompt_render import render


def section(name: str, value: object) -> str:
    return f"{name}:\n{render(value)}\n"


def _verification_view(context: AgentContext) -> object:
    verification = context.latest_verification
    if verification is None:
        return None
    return verification.model_dump(mode="json", exclude={"delta"})


def scientist_sections(context: AgentContext) -> str:
    view = observation_view(
        context.observation, include_raw_frame=not context.compact_observation
    )
    return (
        section("OBSERVATION", view)
        + section("HYPOTHESIS_LEDGER", context.hypothesis_ledger)
        + section("UNRESOLVED_CONTRADICTIONS", context.unresolved_contradictions)
        + section("ACTION_EVIDENCE", context.action_evidence)
    )


def world_modeler_sections(context: AgentContext) -> str:
    return (
        section("LATEST_TRANSITION", context.latest_transition)
        + section("ACTION_EVIDENCE", context.action_evidence)
        + section("WORLD_MODELS", context.world_models)
        + section("LATEST_VERIFICATION", _verification_view(context))
    )


def skeptic_sections(
    context: AgentContext, own_history: tuple[StudentReport, ...]
) -> str:
    return (
        section("WORKING_SCRATCHPAD", context.working_scratchpad)
        + section("UNRESOLVED_CONTRADICTIONS", context.unresolved_contradictions)
        + section("LATEST_VERIFICATION", _verification_view(context))
        + section("OWN_PRIOR_REPORTS", own_history)
    )
