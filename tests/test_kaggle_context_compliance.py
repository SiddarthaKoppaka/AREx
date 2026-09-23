"""Offline deployment and per-game context safeguards."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import JsonValue

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.contracts.cognition import TaskRecord
from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.deployment.kaggle_runtime import build_kaggle_bridge
from arc_agi_3.deployment.kaggle_settings import KaggleSettings
from arc_agi_3.settings import resolve_runtime


class CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "test", "model": "offline"}

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> BackendGeneration:
        del prompt, json_schema
        self.calls += 1
        decision = {
            "assessment": "test action",
            "intent": "observe transition",
            "mode": "execute",
            "action": {"action_id": 1},
            "task_updates": [
                TaskRecord(
                    task_id="p", version=1, purpose="inspect", success_criteria="seen"
                ).model_dump(mode="json")
            ],
        }
        return BackendGeneration(text=json.dumps(decision))


def test_kaggle_profile_is_offline_and_has_generation_room() -> None:
    settings = resolve_runtime("kaggle_submission", environ={})
    assert settings.local_files_only
    assert settings.model_path.is_relative_to("/kaggle/input")
    assert settings.output_dir.is_relative_to("/kaggle/working")
    assert (
        settings.hard_input_limit + settings.max_new_tokens
        <= settings.max_context_tokens
    )


def test_game_context_and_artifacts_are_isolated_with_shared_model(
    tmp_path: Path,
) -> None:
    backend = CountingBackend()
    settings = KaggleSettings(model_path=tmp_path, output_dir=tmp_path / "runs")
    first = build_kaggle_bridge("game-a-v1", settings=settings, backend=backend)
    second = build_kaggle_bridge("game-b-v1", settings=settings, backend=backend)
    assert isinstance(first.session.model, StructuredModelAdapter)
    assert isinstance(second.session.model, StructuredModelAdapter)
    assert first.session.model.backend is backend
    assert second.session.model.backend is backend
    assert first.session.io.workspace is not second.session.io.workspace
    task = TaskRecord(
        task_id="explore", version=1, purpose="learn", success_criteria="seen"
    )
    first.session.io.workspace.tasks.apply_many((task,))
    assert second.session.io.workspace.tasks.current == ()
    assert first.session.io.events.path != second.session.io.events.path


def test_one_callback_action_uses_one_model_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "arc_agi_3.deployment.kaggle_bridge.game_action_from_action",
        lambda action: action,
    )
    backend = CountingBackend()
    settings = KaggleSettings(
        model_path=tmp_path, output_dir=tmp_path / "runs", max_repairs=0
    )
    bridge = build_kaggle_bridge("game-a-v1", settings=settings, backend=backend)
    frame = SimpleNamespace(
        game_id="game-a-v1",
        frame=[[[0]]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1],
        guid="initial",
        full_reset=False,
    )
    assert bridge.choose_action(frame).action_id == 1
    assert backend.calls == 1
    assert bridge.session.io.workspace.tasks.current[0].task_id == "p"
