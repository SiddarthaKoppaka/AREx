"""Faculty-specific, bounded prompts over the one shared AgentContext.

Each specialist sees a subset of the exact same evidence the core agent
would get — built from the same `AgentContext`, never from another
specialist's report (that happens only in the separate review pass below).
"""

from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import CognitiveFaculty
from arc_agi_3.contracts.specialist_report import SpecialistReport

from .specialist_schemas import specialist_report_schema, specialist_review_schema
from .specialist_sections import (
    critic_sections,
    dynamics_sections,
    hypothesis_sections,
    section,
)
from .specialist_text import (
    FACULTY_INSTRUCTIONS,
    SPECIALIST_CONTRACT,
    SPECIALIST_REVIEW_CONTRACT,
)


def specialist_prompt(
    faculty: CognitiveFaculty,
    specialist_id: str,
    context: AgentContext,
    own_history: tuple[SpecialistReport, ...] = (),
) -> str:
    if faculty is CognitiveFaculty.HYPOTHESIS:
        body = hypothesis_sections(context)
    elif faculty is CognitiveFaculty.DYNAMICS:
        body = dynamics_sections(context)
    else:
        body = critic_sections(context, own_history)
    return (
        SPECIALIST_CONTRACT
        + FACULTY_INSTRUCTIONS[faculty.value]
        + f"SPECIALIST_ID: {specialist_id}\nTURN: {context.turn}\n"
        + body
    )


def specialist_review_prompt(
    specialist_id: str,
    faculty: CognitiveFaculty,
    own_report: SpecialistReport,
    other_reports: tuple[SpecialistReport, ...],
    turn: int,
) -> str:
    return (
        SPECIALIST_REVIEW_CONTRACT
        + f"SPECIALIST_ID: {specialist_id}\nFACULTY: {faculty.value}\nTURN: {turn}\n"
        + section("YOUR_PRIOR_REPORT", own_report)
        + section("OTHER_REPORTS", other_reports)
    )


__all__ = [
    "specialist_prompt",
    "specialist_report_schema",
    "specialist_review_prompt",
    "specialist_review_schema",
]
