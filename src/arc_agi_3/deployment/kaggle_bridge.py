"""One-action callback bridge for the official ARC-AGI-3 Agent framework."""

from typing import Any

from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.runtime.session import CognitiveSession

from .kaggle_conversion import game_action_from_action, observation_from_frame


class SessionFinished(RuntimeError):
    """The LM ended the session without authorizing another environment action."""


class KaggleAgentBridge:
    def __init__(self, game_id: str, session: CognitiveSession) -> None:
        self.game_id, self.session = game_id, session

    def ingest(self, latest_frame: Any) -> None:
        observation = observation_from_frame(latest_frame, self.game_id)
        if self.session.observation is None:
            self.session.start(observation)
        elif self.session.pending is not None:
            self.session.observe(observation)

    def is_done(self, latest_frame: Any) -> bool:
        if getattr(latest_frame, "game_id", "") or getattr(latest_frame, "frame", None):
            self.ingest(latest_frame)
        if self.session.result is not None:
            return True
        state = getattr(getattr(latest_frame, "state", None), "value", None)
        if state == EnvironmentState.WIN:
            self.session.next_action()
            return True
        return False

    def choose_action(self, latest_frame: Any) -> Any:
        self.ingest(latest_frame)
        action = self.session.next_action()
        if action is None:
            raise SessionFinished(
                "session ended without an authorized action; the bridge will not "
                "invent a fallback"
            )
        return game_action_from_action(action)


class KaggleAgentMixin:
    """Drop-in callback methods for an official ``agents.agent.Agent`` subclass."""

    arex_bridge: KaggleAgentBridge

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        del frames
        return self.arex_bridge.is_done(latest_frame)

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        del frames
        return self.arex_bridge.choose_action(latest_frame)
