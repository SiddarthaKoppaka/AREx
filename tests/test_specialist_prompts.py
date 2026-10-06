"""Each faculty sees only its own documented sections of the shared context."""

from arc_agi_3.adapters.specialist_prompts import (
    specialist_prompt,
    specialist_review_prompt,
)
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import CognitiveFaculty
from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.testing import FakeLineEnvironment


def context() -> AgentContext:
    return AgentContext(turn=2, observation=FakeLineEnvironment().reset(), budget={})


def test_hypothesis_prompt_has_its_sections_not_dynamics_only_ones() -> None:
    prompt = specialist_prompt(CognitiveFaculty.HYPOTHESIS, "S-hypothesis", context())
    assert "HYPOTHESIS_LEDGER" in prompt and "ACTION_EVIDENCE" in prompt
    assert "WORLD_MODELS" not in prompt
    assert "FACULTY: Hypothesis" in prompt


def test_dynamics_prompt_has_transition_and_world_models() -> None:
    prompt = specialist_prompt(CognitiveFaculty.DYNAMICS, "S-dynamics", context())
    assert "LATEST_TRANSITION" in prompt and "WORLD_MODELS" in prompt
    assert "HYPOTHESIS_LEDGER" not in prompt


def test_critic_prompt_includes_its_own_prior_reports_not_absent() -> None:
    own = (
        SpecialistReport(
            specialist_id="S-critic",
            faculty=CognitiveFaculty.CRITIC,
            turn=1,
            assessment="still unsure",
        ),
    )
    prompt = specialist_prompt(CognitiveFaculty.CRITIC, "S-critic", context(), own)
    assert "OWN_PRIOR_REPORTS" in prompt and "still unsure" in prompt


def test_specialist_review_prompt_shows_others_never_its_own_report() -> None:
    mine = SpecialistReport(
        specialist_id="S-hypothesis",
        faculty=CognitiveFaculty.HYPOTHESIS,
        turn=2,
        assessment="my own claim",
    )
    other = SpecialistReport(
        specialist_id="S-critic",
        faculty=CognitiveFaculty.CRITIC,
        turn=2,
        assessment="other claim",
    )
    prompt = specialist_review_prompt(
        "S-hypothesis", CognitiveFaculty.HYPOTHESIS, mine, (other,), 2
    )
    assert "other claim" in prompt
    assert (
        prompt.count("my own claim") == 1
    )  # only inside YOUR_PRIOR_REPORT, not OTHER_REPORTS
    other_section = prompt[prompt.index("OTHER_REPORTS") :]
    assert "my own claim" not in other_section
