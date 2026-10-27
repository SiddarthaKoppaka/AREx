"""Official-framework Agent subclass; import only where `agents` is installed."""

from typing import Any

from .bridge import DuckBridge
from .factory import make_agent
from .settings import DuckSettings


def build_agent_class(settings: DuckSettings | None = None) -> Any:
    """Return an `agents.agent.Agent` subclass wired to DuckAgent.

    The framework default MAX_ACTIONS is 80, which would end games early.
    """
    from agents.agent import Agent

    resolved = settings or DuckSettings.from_env()

    class DuckKaggleAgent(Agent):  # type: ignore[misc]
        MAX_ACTIONS = resolved.max_actions

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.bridge = DuckBridge(
                self.game_id, make_agent(resolved), resolved.max_actions
            )

        def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
            return self.bridge.is_done(latest_frame)

        def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
            return self.bridge.choose_action(latest_frame)

    return DuckKaggleAgent
