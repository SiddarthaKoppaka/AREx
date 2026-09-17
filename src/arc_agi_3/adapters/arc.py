"""Mechanical adapter for the optional official ARC environment wrapper."""

from typing import Any

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.contracts.observation import Action, Observation


class ArcEnvironmentAdapter:
    def __init__(self, wrapper: Any) -> None:
        self.wrapper = wrapper

    @property
    def metadata(self) -> dict[str, JsonValue]:
        info = getattr(self.wrapper, "info", None)
        if info is None:
            return {"adapter": "arc", "environment": "unknown"}
        values = info.model_dump(mode="json", exclude_none=True)
        return {"adapter": "arc", "info": values}

    def _convert(self, raw: Any) -> Observation:
        if raw is None:
            raise RuntimeError("ARC environment returned no observation")
        frames = [layer.tolist() for layer in raw.frame]
        state = getattr(raw.state, "value", str(raw.state))
        return Observation.build(
            game_id=raw.game_id,
            frame=frames,
            state=EnvironmentState(state),
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            available_actions=raw.available_actions,
            guid=raw.guid,
            full_reset=raw.full_reset,
        )

    def reset(self) -> Observation:
        return self._convert(self.wrapper.reset())

    def step(self, action: Action) -> Observation:
        try:
            from arcengine import GameAction
        except ImportError as error:
            raise RuntimeError("install the 'arc' extra for ARC execution") from error
        game_action = GameAction.from_id(action.action_id)
        return self._convert(self.wrapper.step(game_action, data=dict(action.data)))

    def checkpoint(self) -> dict[str, JsonValue] | None:
        return None

    def restore(self, state: dict[str, JsonValue]) -> Observation:
        raise NotImplementedError("official wrapper does not expose state restore")
