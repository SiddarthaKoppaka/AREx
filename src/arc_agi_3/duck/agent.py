"""Duck-style agent: the model drives a persistent Python REPL until it acts."""

import json
import re

from arc_agi_3.contracts.observation import Action, Observation

from .chat import ChatBackend, Message
from .context import evict
from .dead import DeadClicks
from .fallback import fallback_action
from .history import ActionHistory
from .image import render_png
from .namespace import TurnNamespace
from .prompts import SYSTEM, observation_report
from .render import changes
from .repl import Repl

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
NO_CODE = "No ```python block found. Reply with exactly one block that calls act()."
NO_ACT = "Code ran but act() was not called. Call act() to finish the turn."


class DuckAgent:
    def __init__(
        self,
        chat: ChatBackend,
        *,
        max_inner: int = 4,
        budget_chars: int = 60_000,
        repl_timeout: float = 10.0,
        use_images: bool = False,
        trace_path: str | None = None,
    ) -> None:
        self.trace_path = trace_path
        self.chat, self.max_inner, self.budget = chat, max_inner, budget_chars
        self.use_images = use_images
        self.repl, self.dead = Repl(timeout=repl_timeout), DeadClicks()
        self.history = ActionHistory()
        self.turn = TurnNamespace(self.repl.namespace, self.dead)
        self.messages: list[Message] = [{"role": "system", "content": SYSTEM}]
        self.prev: list[list[int]] | None = None
        self.model_calls = self.fallbacks = 0

    def decide(self, obs: Observation) -> Action:
        grid = [list(row) for row in obs.frame[-1]] if obs.frame else self.prev
        if not grid:  # no frame yet and none remembered: nothing to reason about
            return Action(
                action_id=0 if 0 in obs.available_actions else obs.available_actions[0]
            )
        total, _ = changes(self.prev, grid)
        self.dead.observed(total, obs.levels_completed)
        self.history.observed(total)
        self.turn.begin(obs, grid, self.prev)
        self.prev = grid
        report = observation_report(obs, self.dead.dead, self.history.note())
        self._say("user", report, render_png(grid) if self.use_images else None)
        action = self._act() or self._fallback(obs, grid)
        self.dead.decided(grid, action)
        self.history.decided(action)
        self._trace({"action": action.model_dump(exclude={"schema_version"})})
        return action

    def _act(self) -> Action | None:
        for _ in range(self.max_inner):
            reply = self.chat.chat(self.messages)
            self.model_calls += 1
            self._say("assistant", reply)
            match = CODE_BLOCK.search(reply)
            output = self.repl.run(match.group(1)) if match else NO_CODE
            if self.turn.chosen is not None:
                return self.turn.chosen
            self._say("user", f"OUTPUT:\n{output}\n" + (NO_ACT if match else ""))
        return None

    def _fallback(self, obs: Observation, grid: list[list[int]]) -> Action:
        self.fallbacks += 1
        return fallback_action(obs, grid, self.dead.dead)

    def _trace(self, record: dict[str, object]) -> None:
        if self.trace_path:
            with open(self.trace_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def _say(self, role: str, content: str, image: bytes | None = None) -> None:
        message: Message = {"role": role, "content": content}
        if role == "assistant":
            self._trace({"reply": content})
        if image is not None:
            message["image"] = image
        self.messages = evict([*self.messages, message], self.budget)
