"""Official ARC-AGI-3 Agent wrapper around the AREx Kaggle bridge."""

from typing import Any

from agents.agent import Agent

from arc_agi_3.deployment import KaggleAgentMixin, KaggleSettings, build_kaggle_bridge


class MyAgent(KaggleAgentMixin, Agent):
    MAX_ACTIONS = 40

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        settings = KaggleSettings.from_env()
        self.MAX_ACTIONS = settings.max_actions
        self.arex_bridge = build_kaggle_bridge(self.game_id, settings=settings)
