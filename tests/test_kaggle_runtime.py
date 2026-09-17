"""Kaggle composition creates isolated sessions over a shared backend."""

import json
from pathlib import Path

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.deployment.kaggle_runtime import build_kaggle_bridge
from arc_agi_3.deployment.kaggle_settings import KaggleSettings


class Backend:
    metadata = {"provider": "test", "model": "qwen"}

    def generate(self, prompt, json_schema):  # type: ignore[no-untyped-def]
        del prompt, json_schema
        return BackendGeneration(text=json.dumps({"mode": "stop"}))


def test_builds_fresh_game_session_with_competition_budgets(tmp_path: Path) -> None:
    settings = KaggleSettings(
        model_path=tmp_path,
        output_dir=tmp_path / "runs",
        max_actions=7,
        max_model_calls=2,
    )
    bridge = build_kaggle_bridge("ls20", settings=settings, backend=Backend())
    assert bridge.game_id == "ls20"
    assert bridge.session.io.config.max_turns == 7
    assert bridge.session.io.config.budget.limits["model_calls"] == 2
    assert bridge.session.io.environment.metadata["owns_environment_loop"] is False
