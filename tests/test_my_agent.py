"""The starter-facing MyAgent remains a thin official-framework adapter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import arc_agi_3.deployment as deployment
from arc_agi_3.deployment import KaggleSettings


class FakeAgent:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args
        self.game_id = kwargs["game_id"]


def test_my_agent_builds_bridge_and_delegates(tmp_path: Path, monkeypatch: Any) -> None:
    agents = ModuleType("agents")
    agent_module = ModuleType("agents.agent")
    agent_module.Agent = FakeAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.agent", agent_module)
    bridge = SimpleNamespace(
        is_done=lambda frame: frame == "done",
        choose_action=lambda frame: ("action", frame),
    )
    settings = KaggleSettings(
        model_path=tmp_path, output_dir=tmp_path, max_actions=9, require_gpu=False
    )
    monkeypatch.setattr(KaggleSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(deployment, "build_kaggle_bridge", lambda *args, **kw: bridge)
    path = Path("kaggle/agent/my_agent.py")
    spec = importlib.util.spec_from_file_location("test_submission_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.MyAgent(game_id="ls20")
    assert instance.MAX_ACTIONS == 9
    assert instance.is_done([], "done") is True
    assert instance.choose_action([], "frame") == ("action", "frame")
