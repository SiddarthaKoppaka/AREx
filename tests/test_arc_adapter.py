"""Optional smoke test against the bundled local ARC environment."""

from pathlib import Path

import pytest

from arc_agi_3.adapters.arc import ArcEnvironmentAdapter
from arc_agi_3.contracts.observation import Action

arc_agi = pytest.importorskip("arc_agi")


def test_bundled_environment_reset_and_step(tmp_path: Path) -> None:
    root = Path("arc-prize-2026-arc-agi-3/environment_files")
    if not root.exists():
        pytest.skip("bundled ARC environments are unavailable")
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(root),
        recordings_dir=str(tmp_path),
    )
    wrapper = arcade.make("ls20", seed=0, save_recording=False)
    assert wrapper is not None
    adapter = ArcEnvironmentAdapter(wrapper)
    initial = adapter.reset()
    action_id = next(value for value in initial.available_actions if value != 0)
    after = adapter.step(Action(action_id=action_id))
    assert initial.game_id.startswith("ls20-")
    assert initial.observation_hash != after.observation_hash
    assert adapter.metadata["adapter"] == "arc"
