"""Reki's dead-signature rule: stop clicking object types that never respond."""

from arc_agi_3.contracts.observation import Action

from .render import Grid
from .segment import Signature, signature_at


class DeadClicks:
    """Signatures whose click changed nothing; cleared when a level is cleared."""

    def __init__(self) -> None:
        self.dead: set[Signature] = set()
        self.level = 0
        self.pending: Signature | None = None

    def observed(self, changed_cells: int, level: int) -> None:
        if level != self.level:
            self.level = level
            self.dead.clear()
        elif self.pending is not None and changed_cells == 0:
            self.dead.add(self.pending)
        self.pending = None

    def decided(self, grid: Grid, action: Action) -> None:
        if action.action_id != 6:
            return
        x, y = int(action.data["x"]), int(action.data["y"])
        try:
            self.pending = signature_at(grid, x, y)
        except IndexError:  # click outside the rendered grid
            self.pending = None
