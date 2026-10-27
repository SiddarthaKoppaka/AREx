"""Feedback on the agent's own recent behaviour: no-op and repeated actions."""

from arc_agi_3.contracts.observation import Action

REPEAT_WARN = 3


def label(action: Action) -> str:
    if action.action_id == 6:
        return f"ACTION6({action.data['x']},{action.data['y']})"
    return f"ACTION{action.action_id}" if action.action_id else "RESET"


class ActionHistory:
    """Remembers what each action did so the status line can say so."""

    def __init__(self) -> None:
        self.last: str | None = None
        self.changed: int | None = None
        self.repeat = 0
        self.idle = 0

    def observed(self, changed_cells: int) -> None:
        self.changed = changed_cells
        self.idle = self.idle + 1 if changed_cells == 0 else 0

    def decided(self, action: Action) -> None:
        name = label(action)
        self.repeat = self.repeat + 1 if name == self.last else 1
        self.last, self.changed = name, None

    def note(self) -> str:
        if self.last is None or self.changed is None:
            return ""
        effect = "NO EFFECT" if self.changed == 0 else f"changed {self.changed} cells"
        lines = [f"LAST_ACTION: {self.last} -> {effect}"]
        if self.repeat >= REPEAT_WARN:
            lines.append(f"WARNING: {self.last} repeated {self.repeat} times in a row.")
        if self.idle >= REPEAT_WARN:
            lines.append(f"WARNING: last {self.idle} actions changed nothing.")
        return "\n".join(lines)
