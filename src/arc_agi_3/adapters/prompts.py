"""Minimal public-workspace prompt projection."""

from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.trace.canonical import canonical_json


def _ordered_context(context: AgentContext) -> str:
    sections = (
        ("CURRENT_OBSERVATION", context.observation),
        ("VERIFIED_WORLD_MODELS", context.world_models),
        ("WORKING_SCRATCHPAD", context.working_scratchpad),
        ("RECENT_TURNS", context.recent_events),
        ("RELEVANT_EPISODES", context.episodic_memory),
        (
            "OTHER_CONTEXT",
            {
                "turn": context.turn,
                "budget": context.budget,
                "recent_event_refs": context.recent_event_refs,
                "hypotheses": context.hypotheses,
                "tasks": context.tasks,
                "recovery_evidence": context.recovery_evidence,
            },
        ),
    )
    return "\n".join(f"{name}:\n{canonical_json(value)}" for name, value in sections)


def decision_prompt(
    context: AgentContext,
    *,
    prior_output: str | None = None,
    validation_error: str | None = None,
) -> str:
    base = (
        "You are the agent. Return exactly one CognitiveDecision as raw JSON "
        "matching the provided schema. Do not use Markdown code fences, XML tags, "
        "commentary, or text outside the JSON object. "
        "Own all semantic interpretation, intent, strategy, and action choice. "
        "Do not provide private chain-of-thought or hidden reasoning. Use assessment, "
        "intent, "
        "considered_options, decision_summary, observation_summary, and "
        "expected_result only for concise public decision rationale and summaries. "
        "Structural invariants: execute mode must "
        "set exactly one of action or action_chunk; every other mode must set "
        "both action and action_chunk to null. Recover mode must set recovery; "
        "every other mode must set recovery to null. Keep decision-history fields "
        "concise and public; never include private chain-of-thought. Update working "
        "memory only through scratchpad_updates. Verified facts must cite existing "
        "prior event IDs in evidence_refs; capabilities are harness-owned. Use "
        "hypothesis_proposals and hypothesis_updates to add or reject hypotheses.\n"
        "To inspect older exact events, use tool_requests with tool_name "
        'retrieve_events and arguments such as {"event_ids":["event-id"],'
        '"limit":1}.\nCONTEXT:\n' + _ordered_context(context)
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
