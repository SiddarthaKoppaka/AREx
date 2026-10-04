"""Bounded working-set prompt projection for the primary cognitive generation.

Repair generations use `repair_prompt.build_repair_prompt` instead and never
pass through here: this prompt is cognition, not format repair.
"""

from typing import Any

from arc_agi_3.contracts.decision import AgentContext

from .episode_projection import episode_prompt_view
from .observation_projection import observation_view
from .prompt_render import render
from .prompt_text import CONTRACT, EPISTEMICS, TOOLS
from .task_projection import task_prompt_view
from .tool_schemas import tool_schemas


def _sections(context: AgentContext) -> tuple[tuple[str, Any], ...]:
    verification = context.latest_verification
    return (
        (
            "CURRENT_OBSERVATION",
            observation_view(
                context.observation, include_raw_frame=not context.compact_observation
            ),
        ),
        ("CURRENT_OBSERVATION_EVENT_ID", context.current_observation_event_id),
        (
            "LATEST_TRANSITION",
            {
                "event_id": context.latest_transition_event_id,
                "evidence": context.latest_transition,
            },
        ),
        (
            "LATEST_VERIFICATION",
            verification.model_dump(mode="json", exclude={"delta"})
            if verification
            else None,
        ),
        ("HYPOTHESIS_LEDGER", context.hypothesis_ledger),
        ("UNRESOLVED_CONTRADICTIONS", context.unresolved_contradictions),
        ("ACTION_EVIDENCE", context.action_evidence),
        ("WORKING_SCRATCHPAD", context.working_scratchpad),
        ("TASK_GRAPH", task_prompt_view(context.tasks)),
        ("MODEL_AUTHORED_WORLD_MODELS", context.world_models),
        ("RECENT_TURNS", context.recent_events),
        ("RELEVANT_EPISODES", episode_prompt_view(context.episodic_memory)),
        (
            "OTHER_CONTEXT",
            {
                "turn": context.turn,
                "budget": context.budget,
                "context_stats": context.context_stats,
                "recovery_evidence": context.recovery_evidence,
            },
        ),
    )


def decision_prompt(context: AgentContext) -> str:
    body = "\n".join(f"{name}:\n{render(value)}" for name, value in _sections(context))
    contracts = f"TOOL_CONTRACTS:\n{render(tool_schemas())}\n"
    return CONTRACT + EPISTEMICS + TOOLS + contracts + "CONTEXT:\n" + body
