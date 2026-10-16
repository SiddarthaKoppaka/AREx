"""PR-12 (idea A): `untried_actions_from_current_state` gives a direct
answer to "what hasn't been tried from here," instead of requiring the
model to scan `action_evidence`'s per-action table and check
`attempts_from_current_state == 0` on each row itself."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.evidence.action_cues import untried_actions_from_current_state
from arc_agi_3.evidence.history import TransitionRecord
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def _observation() -> Observation:
    return Observation.build(
        game_id="g",
        frame=[[[0]]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=(1, 2, 3, 6, 7),
    )


def _click_records(tmp_path: Path, observation: Observation, count: int):
    config = RunConfig(run_id="cues", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    records = []
    for index in range(count):
        action_event = runner.io.append(
            EventType.ACTION,
            "executor",
            0,
            {
                "action": {"action_id": 6, "data": {"x": index, "y": index}},
                "before_hash": observation.observation_hash,
            },
            (),
        )
        records.append(
            TransitionRecord(action_event, action_event, None, action_event, None)
        )
    return records


def test_actions_never_tried_from_this_state_are_reported(tmp_path: Path) -> None:
    observation = _observation()
    records = _click_records(tmp_path, observation, 1)
    untried = untried_actions_from_current_state(records, observation)
    assert set(untried) == {1, 2, 3, 7}


def test_action_six_is_excluded_even_though_it_was_never_tried(
    tmp_path: Path,
) -> None:
    observation = _observation()
    untried = untried_actions_from_current_state([], observation)
    assert 6 not in untried
    assert set(untried) == {1, 2, 3, 7}


def test_many_click_variants_from_a_different_state_do_not_count_as_tried(
    tmp_path: Path,
) -> None:
    observation = _observation()
    other = Observation.build(
        game_id="g",
        frame=[[[1]]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=(1, 2, 3, 6, 7),
    )
    records = _click_records(tmp_path, other, 9)
    untried = untried_actions_from_current_state(records, observation)
    assert set(untried) == {1, 2, 3, 7}
