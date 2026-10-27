"""Prompts: a short generic system message and a per-turn status report."""

from arc_agi_3.contracts.observation import Observation

from .segment import Signature

SYSTEM = """You play an unknown grid game (ARC-AGI-3). Nothing is explained: \
infer the rules by experimenting, then win levels with as few actions as possible.

Each turn you get a status line (and usually an image of the frame). The frame \
is NOT written out as text: read it with Python. Reply with ONE ```python block; \
the REPL is persistent (variables survive).
Variables: grid (list of rows, hex-digit colors 0-15), prev_grid, frames, state, \
levels_completed, win_levels, available_actions, arr (numpy, if installed).
Helpers: act(action_id, x=None, y=None) ends your turn with that action \
(ACTION6 needs x,y in 0-63; 0 is RESET); diff(prev_grid, grid) prints what \
changed, grouped by color change with bounding boxes; components(grid); \
zoom(grid, top, left, bottom, right); click_candidates(grid) (skips types \
whose click did nothing); changes(prev_grid, grid); print() to inspect.
Call act() exactly once per turn, after you have inspected the frame in code.

Method: the action meanings are unknown. In your first turns try each available \
action once, then diff() to see what moved, and record it in a dict `effects` \
(e.g. effects[1] = "player up 5px"). Never repeat an action that changed \
nothing; when stuck, test a different action. Form a guess about the goal and \
check it against diff(). Keep `effects` and your goal guess in variables or \
comments: old messages are forgotten."""


def observation_report(
    obs: Observation,
    dead: set[Signature] = frozenset(),  # type: ignore[assignment]
    history: str = "",
) -> str:
    note = f"\nDEAD_CLICK_TYPES (color,size): {sorted(dead)}" if dead else ""
    past = f"\n{history}" if history else ""
    return (
        f"STATE: {obs.state.value}  LEVELS: {obs.levels_completed}/{obs.win_levels}"
        f"  AVAILABLE_ACTIONS: {list(obs.available_actions)}{note}{past}\n"
        "Frame is in `grid` (previous frame in `prev_grid`)."
    )
