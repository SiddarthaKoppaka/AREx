"""Exploration values are arithmetic evidence, never an automatic policy."""

import pytest

from arc_agi_3.exploration import (
    ExplorationHistory,
    ExplorationRequest,
    ProbeEstimate,
    evaluate_exploration,
)
from arc_agi_3.exploration.contracts import ExplorationWeights


def request() -> ExplorationRequest:
    return ExplorationRequest(
        request_id="value",
        probes=(
            ProbeEstimate(
                probe_id="a",
                candidate_state_hash="seen",
                drig=2,
                expected_reward_progress=1,
                hypothesis_discrimination=2,
                action_cost=1,
                inference_cost=0.5,
                risk_cost=0.25,
                repeat_cost=0.25,
                repeat_cost_per_visit=1,
            ),
            ProbeEstimate(
                probe_id="b",
                candidate_state_hash="new",
                drig=1,
                expected_reward_progress=0,
                hypothesis_discrimination=0,
                action_cost=0,
                inference_cost=0,
                risk_cost=0,
                repeat_cost=0,
            ),
        ),
        weights=ExplorationWeights(
            information=2, reward_progress=1, hypothesis_discrimination=0.5
        ),
        executable_plan_utility=3,
        safety_margin=0.5,
        recent_marginal_values=(0.2, 0.1, 0.0),
        saturation_threshold=0.2,
    )


def test_values_include_exact_history_cost_and_saturation() -> None:
    history = ExplorationHistory()
    history.record("seen", 0.2)
    result = evaluate_exploration(request(), history)
    scores = {item.probe_id: item for item in result.scores}
    assert scores["a"].value == pytest.approx(3.0)
    assert scores["a"].prior_visits == 1
    assert scores["b"].novel_state
    assert result.maximizing_probe_ids == ("a",)
    assert result.saturated
    assert not result.exploration_dominated


def test_plan_comparison_is_reported_without_selecting_an_action() -> None:
    updated = request().model_copy(update={"executable_plan_utility": 5.0})
    result = evaluate_exploration(updated)
    assert result.exploration_dominated
    assert all(not hasattr(score, "action") for score in result.scores)
