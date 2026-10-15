"""Turn a validated skill into a normal ActionChunk, or refuse with a
reason - never a second execution mechanism, never a silent downgrade."""

from dataclasses import dataclass

from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.execution import ActionChunk

from .contracts import Skill
from .preconditions import check_precondition, is_eligible_for_game


@dataclass(frozen=True)
class SkillInstantiation:
    chunk: ActionChunk | None
    refusal_reason: str | None

    @property
    def applicable(self) -> bool:
        return self.chunk is not None


def instantiate_skill(
    skill: Skill, context: AgentContext, game_id: str
) -> SkillInstantiation:
    if not is_eligible_for_game(skill, game_id):
        return SkillInstantiation(
            None, f"skill {skill.skill_id!r} is not eligible for game {game_id!r}"
        )
    reason = check_precondition(skill, context)
    if reason is not None:
        return SkillInstantiation(None, reason)
    return SkillInstantiation(skill.chunk_template, None)
