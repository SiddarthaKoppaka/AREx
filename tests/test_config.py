"""Configuration identity excludes artifact location, not semantics."""

from pathlib import Path

from arc_agi_3.config import RunConfig


def test_output_directory_does_not_change_semantic_hash() -> None:
    left = RunConfig(run_id="run", experiment_id="experiment", output_dir=Path("one"))
    right = left.model_copy(update={"output_dir": Path("two")})
    assert left.config_hash == right.config_hash


def test_budget_change_changes_semantic_hash() -> None:
    original = RunConfig(run_id="run", experiment_id="experiment")
    changed = original.model_copy(update={"max_turns": 99})
    assert original.config_hash != changed.config_hash
