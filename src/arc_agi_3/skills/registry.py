"""Versioned skill storage, mirroring `world_model/store.py`'s shape."""

from .contracts import Skill


class SkillLibrary:
    def __init__(self) -> None:
        self._history: dict[str, list[Skill]] = {}

    def add(self, skill: Skill) -> None:
        history = self._history.setdefault(skill.skill_id, [])
        if any(item.version == skill.version for item in history):
            raise ValueError(
                f"version {skill.version} already exists for {skill.skill_id!r}"
            )
        expected = 1 if not history else history[-1].version + 1
        if skill.version != expected:
            raise ValueError(f"skill {skill.skill_id!r} requires version {expected}")
        history.append(skill)

    def skill_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._history))

    def latest(self, skill_id: str) -> Skill:
        history = self._history.get(skill_id)
        if not history:
            raise KeyError(f"unknown skill {skill_id!r}")
        return history[-1]

    def for_game(self, game_id: str) -> tuple[Skill, ...]:
        from .preconditions import is_eligible_for_game

        return tuple(
            history[-1]
            for history in self._history.values()
            if is_eligible_for_game(history[-1], game_id)
        )
