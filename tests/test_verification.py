"""Mechanical deltas never infer semantic consequences."""

from arc_agi_3.contracts.decision import ExpectedOutcome
from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.verification import compare_observations, verify_outcome


def observed(position: int, state: EnvironmentState) -> Observation:
    row = [0, 0, 0]
    row[position] = 2
    return Observation.build(
        game_id="test",
        frame=[[row]],
        state=state,
        levels_completed=int(state is EnvironmentState.WIN),
        win_levels=1,
        available_actions=[] if state is EnvironmentState.WIN else [1],
    )


def test_exact_delta_counts_changed_cells() -> None:
    delta = compare_observations(
        observed(0, EnvironmentState.NOT_FINISHED),
        observed(1, EnvironmentState.NOT_FINISHED),
    )
    assert delta.changed_cells == 2
    assert delta.metadata_changes == {}


def test_failed_prediction_returns_evidence_only() -> None:
    result = verify_outcome(
        ExpectedOutcome(state=EnvironmentState.WIN),
        observed(0, EnvironmentState.NOT_FINISHED),
        observed(1, EnvironmentState.NOT_FINISHED),
    )
    assert not result.passed
    assert result.mismatches == ("state",)
