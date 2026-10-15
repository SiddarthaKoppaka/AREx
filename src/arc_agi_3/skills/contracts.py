"""Versioned, reusable action macros. See PR-09-report.md and the
existing per-game-isolation invariant (handoff.md's "no leakage of
inferred hidden game dynamics across isolated games... General skills
must declare eligibility and be revalidated on unfamiliar versions").

A `Skill` never executes anything itself - `instantiate()` only ever
produces a normal `ActionChunk`, which goes through the existing,
unmodified per-step authorization/verification/interrupt path
(`runtime/session_chunk.py`). There is no second execution mechanism to
keep in sync with the first.
"""

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.execution import ActionChunk


class SkillPrecondition(Contract):
    """What must already be true for this skill to be considered
    applicable. Checked against the current decision context, never
    against a hardcoded game solution."""

    required_hypothesis_ids: tuple[str, ...] = ()
    required_verified_fact_ids: tuple[str, ...] = ()
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Skill(Contract):
    skill_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    game_id: str = Field(min_length=1)
    transfer_eligible: bool = False
    chunk_template: ActionChunk
    precondition: SkillPrecondition
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    success_case_refs: tuple[str, ...] = ()
    failure_case_refs: tuple[str, ...] = ()
    model_version_dependencies: tuple[str, ...] = ()
