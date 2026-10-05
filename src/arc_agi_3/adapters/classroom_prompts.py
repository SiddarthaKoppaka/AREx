"""Role-specific, bounded prompts over the one shared AgentContext.

Each Student sees a subset of the exact same evidence the Teacher would
get — built from the same `AgentContext`, never from another Student's
report (that happens only in the separate peer-review pass below).
"""

from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import StudentRole

from .classroom_schemas import peer_review_schema, student_report_schema
from .classroom_sections import (
    scientist_sections,
    section,
    skeptic_sections,
    world_modeler_sections,
)
from .classroom_text import PEER_REVIEW_CONTRACT, ROLE_INSTRUCTIONS, STUDENT_CONTRACT


def student_prompt(
    role: StudentRole,
    student_id: str,
    context: AgentContext,
    own_history: tuple[StudentReport, ...] = (),
) -> str:
    if role is StudentRole.SCIENTIST:
        body = scientist_sections(context)
    elif role is StudentRole.WORLD_MODELER:
        body = world_modeler_sections(context)
    else:
        body = skeptic_sections(context, own_history)
    return (
        STUDENT_CONTRACT
        + ROLE_INSTRUCTIONS[role.value]
        + f"STUDENT_ID: {student_id}\nTURN: {context.turn}\n"
        + body
    )


def peer_review_prompt(
    student_id: str,
    role: StudentRole,
    own_report: StudentReport,
    peer_reports: tuple[StudentReport, ...],
    turn: int,
) -> str:
    return (
        PEER_REVIEW_CONTRACT
        + f"STUDENT_ID: {student_id}\nROLE: {role.value}\nTURN: {turn}\n"
        + section("YOUR_PRIOR_REPORT", own_report)
        + section("PEER_REPORTS", peer_reports)
    )


__all__ = [
    "peer_review_prompt",
    "peer_review_schema",
    "student_prompt",
    "student_report_schema",
]
