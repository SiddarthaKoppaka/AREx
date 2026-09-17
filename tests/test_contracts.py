"""Contract validation and stable mechanical identity."""

import pytest
from pydantic import ValidationError

from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EnvironmentState
from arc_agi_3.contracts.observation import Action, Observation


def observation() -> Observation:
    return Observation.build(
        game_id="test",
        frame=[[[1, 0]]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1],
    )


def test_observation_hash_is_stable() -> None:
    assert observation().observation_hash == observation().observation_hash


def test_execute_requires_model_selected_action() -> None:
    with pytest.raises(ValidationError):
        CognitiveDecision(
            assessment="public summary",
            intent="move",
            mode=DecisionMode.EXECUTE,
        )


def test_non_execute_cannot_smuggle_action() -> None:
    with pytest.raises(ValidationError):
        CognitiveDecision(
            assessment="public summary",
            intent="inspect",
            mode=DecisionMode.INVESTIGATE,
            action=Action(action_id=1),
        )


def test_action6_requires_competition_grid_coordinates() -> None:
    assert Action(action_id=6, data={"x": 0, "y": 63}).data == {"x": 0, "y": 63}
    with pytest.raises(ValueError, match="x"):
        Action(action_id=6, data={"x": 64, "y": 0})
