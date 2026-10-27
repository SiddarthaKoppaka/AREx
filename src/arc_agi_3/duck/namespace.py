"""REPL variables and helpers refreshed each turn; `act` records the choice."""

from typing import Any

from arc_agi_3.contracts.observation import Action, Observation

from .dead import DeadClicks
from .render import Grid, changes, describe_diff
from .segment import click_candidates, components, zoom


class TurnNamespace:
    """Owns the REPL namespace and the single action chosen during a turn."""

    def __init__(self, namespace: dict[str, Any], dead: DeadClicks) -> None:
        self.namespace = namespace
        self.chosen: Action | None = None
        self.available: tuple[int, ...] = ()
        namespace.update(
            act=self.act,
            components=components,
            zoom=zoom,
            click_candidates=lambda g, limit=8: click_candidates(
                g, limit, exclude=dead.dead
            ),
            changes=changes,
            diff=lambda prev, cur: print(describe_diff(prev, cur)),
        )

    def begin(self, obs: Observation, grid: Grid, prev: Grid | None) -> None:
        self.chosen, self.available = None, obs.available_actions
        self.namespace.update(
            grid=grid,
            prev_grid=prev,
            frames=[[list(r) for r in layer] for layer in obs.frame],
            state=obs.state.value,
            levels_completed=obs.levels_completed,
            win_levels=obs.win_levels,
            available_actions=list(obs.available_actions),
        )
        try:
            import numpy

            self.namespace.update(np=numpy, arr=numpy.array(grid))
        except ImportError:
            pass

    def act(self, action_id: int, x: int | None = None, y: int | None = None) -> None:
        if self.chosen is not None:
            raise ValueError("act() already called this turn")
        if action_id not in self.available:
            raise ValueError(f"action {action_id} not in {list(self.available)}")
        data = {"x": x, "y": y} if action_id == 6 else {}
        self.chosen = Action.model_validate({"action_id": action_id, "data": data})
