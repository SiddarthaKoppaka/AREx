"""Reusable, validated action macros. See PR-09-report.md.

Not wired into the live decision loop or any LM-facing tool yet - this
is the contract, precondition checking, registry, and instantiation
logic, proven correct by tests, consistent with how PR-03/04/05 each
landed their contracts before any live integration.
"""

from .contracts import Skill, SkillPrecondition
from .instantiate import SkillInstantiation, instantiate_skill
from .preconditions import check_precondition, is_eligible_for_game
from .registry import SkillLibrary

__all__ = [
    "Skill",
    "SkillInstantiation",
    "SkillLibrary",
    "SkillPrecondition",
    "check_precondition",
    "instantiate_skill",
    "is_eligible_for_game",
]
