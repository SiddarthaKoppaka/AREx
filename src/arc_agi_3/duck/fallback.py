"""Legal action used only when the model produced no valid act() call."""

from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.contracts.observation import Action, Observation

from .render import Grid
from .segment import Signature, click_candidates


def fallback_action(
    obs: Observation,
    grid: Grid,
    dead: set[Signature] | frozenset[Signature] = frozenset(),
) -> Action:
    legal = obs.available_actions
    if obs.state is EnvironmentState.GAME_OVER or not legal:
        return Action(action_id=0)
    action_id = next((a for a in legal if a != 0), legal[0])
    if action_id != 6:
        return Action(action_id=action_id)
    x, y = (click_candidates(grid, 1, exclude=dead) or [(32, 32)])[0]
    return Action(action_id=6, data={"x": x, "y": y})
