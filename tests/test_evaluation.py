"""RHAE interpretation and deterministic aggregation."""

import pytest

from arc_agi_3.evaluation import environment_score, level_score


def test_level_score_caps_final_squared_ratio() -> None:
    assert level_score(20, 2) == 1.15
    assert level_score(10, 20) == 0.25


def test_environment_cap_rewards_completion() -> None:
    assert environment_score((1.15, 0.0), completed=1) == pytest.approx(1 / 3)
    assert environment_score((1.0, 1.0), completed=2) == 1.0


def test_action_counts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        level_score(10, 0)
