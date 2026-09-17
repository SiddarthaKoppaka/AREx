"""Mechanical environment records without semantic interpretation."""

from collections.abc import Sequence

from pydantic import Field

from arc_agi_3.trace.canonical import canonical_hash

from .base import Contract
from .enums import EnvironmentState

Frame = tuple[tuple[tuple[int, ...], ...], ...]


class Action(Contract):
    action_id: int = Field(ge=0, le=7)
    data: dict[str, int | float | str | bool] = Field(default_factory=dict)


class Observation(Contract):
    game_id: str
    frame: Frame
    state: EnvironmentState
    levels_completed: int = Field(ge=0)
    win_levels: int = Field(ge=0)
    available_actions: tuple[int, ...]
    observation_hash: str
    guid: str | None = None
    full_reset: bool = False

    @classmethod
    def build(
        cls,
        *,
        game_id: str,
        frame: Sequence[Sequence[Sequence[int]]],
        state: EnvironmentState | str,
        levels_completed: int,
        win_levels: int,
        available_actions: Sequence[int],
        guid: str | None = None,
        full_reset: bool = False,
    ) -> "Observation":
        frozen = tuple(tuple(tuple(row) for row in layer) for layer in frame)
        mechanical = {
            "frame": frozen,
            "state": str(state),
            "levels_completed": levels_completed,
            "win_levels": win_levels,
            "available_actions": tuple(available_actions),
            "full_reset": full_reset,
        }
        resolved_state = EnvironmentState(state)
        return cls(
            game_id=game_id,
            frame=frozen,
            state=resolved_state,
            levels_completed=levels_completed,
            win_levels=win_levels,
            available_actions=tuple(available_actions),
            observation_hash=canonical_hash(mechanical),
            guid=guid,
            full_reset=full_reset,
        )
