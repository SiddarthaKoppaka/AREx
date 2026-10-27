"""Official-framework callbacks for DuckAgent (mirrors KaggleAgentBridge)."""

from typing import Any

from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.deployment.kaggle_conversion import (
    game_action_from_action,
    observation_from_frame,
)

from .agent import DuckAgent


class DuckBridge:
    def __init__(self, game_id: str, agent: DuckAgent, max_actions: int = 2000) -> None:
        self.game_id, self.agent, self.max_actions = game_id, agent, max_actions
        self.actions_taken = 0

    def is_done(self, latest_frame: Any) -> bool:
        state = getattr(getattr(latest_frame, "state", None), "value", None)
        won = state == EnvironmentState.WIN
        return won or self.actions_taken >= self.max_actions

    def choose_action(self, latest_frame: Any) -> Any:
        obs = observation_from_frame(latest_frame, self.game_id)
        self.actions_taken += 1
        return game_action_from_action(self.agent.decide(obs))
