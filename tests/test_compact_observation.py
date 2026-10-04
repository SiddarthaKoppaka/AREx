"""The default prompt and observation view are compact and non-semantic."""

from arc_agi_3.adapters.observation_projection import observation_view
from arc_agi_3.adapters.prompts import decision_prompt
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.contracts.observation import Observation

SEMANTIC_WORDS = ("player", "goal", "wall", "hazard", "enemy")


def big_observation() -> Observation:
    return Observation.build(
        game_id="big",
        frame=[[[(row + col) % 4 for col in range(16)] for row in range(16)]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1],
    )


def test_compact_view_has_no_semantic_labels_and_no_raw_grid() -> None:
    view = observation_view(big_observation())
    assert "frame_view" not in view
    assert set(view) >= {"observation_hash", "mechanical_summary"}
    text = str(view).lower()
    assert all(word not in text for word in SEMANTIC_WORDS)


def test_include_raw_frame_opt_in_still_embeds_the_grid() -> None:
    view = observation_view(big_observation(), include_raw_frame=True)
    assert "frame_view" in view


def test_default_prompt_omits_the_raw_grid_but_keeps_mechanical_summary() -> None:
    context = AgentContext(turn=1, observation=big_observation(), budget={})
    prompt = decision_prompt(context)
    assert '"frame_view"' not in prompt
    assert '"dimensions"' in prompt
    start = prompt.index("CURRENT_OBSERVATION:\n")
    section = prompt[start : prompt.index("\nCURRENT_OBSERVATION_EVENT_ID", start)]
    assert all(word not in section.lower() for word in SEMANTIC_WORDS)
