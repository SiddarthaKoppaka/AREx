"""Mechanical conversion at the official ARC agent callback boundary."""

from typing import Any

from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.contracts.observation import Action, Observation


def observation_from_frame(frame: Any, game_id: str) -> Observation:
    layers = getattr(frame, "frame", ())
    frozen_layers = [
        layer.tolist() if hasattr(layer, "tolist") else layer for layer in layers
    ]
    raw_state = getattr(frame, "state", EnvironmentState.NOT_PLAYED)
    state = getattr(raw_state, "value", str(raw_state))
    actions = [int(getattr(item, "value", item)) for item in frame.available_actions]
    return Observation.build(
        game_id=getattr(frame, "game_id", "") or game_id,
        frame=frozen_layers,
        state=state,
        levels_completed=frame.levels_completed,
        win_levels=frame.win_levels,
        available_actions=actions,
        guid=getattr(frame, "guid", None),
        full_reset=getattr(frame, "full_reset", False),
    )


def game_action_from_action(action: Action) -> Any:
    try:
        from arcengine import GameAction
    except ImportError as error:
        raise RuntimeError("install the 'arc' extra for Kaggle execution") from error
    game_action = GameAction.from_id(action.action_id)
    game_action.set_data(dict(action.data))
    return game_action
