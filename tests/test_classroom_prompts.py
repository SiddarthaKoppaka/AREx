"""Each role sees only its own documented sections of the shared context."""

from arc_agi_3.adapters.classroom_prompts import peer_review_prompt, student_prompt
from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import StudentRole
from arc_agi_3.testing import FakeLineEnvironment


def context() -> AgentContext:
    return AgentContext(turn=2, observation=FakeLineEnvironment().reset(), budget={})


def test_scientist_prompt_has_its_sections_not_world_modeler_only_ones() -> None:
    prompt = student_prompt(StudentRole.SCIENTIST, "S-scientist", context())
    assert "HYPOTHESIS_LEDGER" in prompt and "ACTION_EVIDENCE" in prompt
    assert "WORLD_MODELS" not in prompt
    assert "ROLE: Scientist" in prompt


def test_world_modeler_prompt_has_transition_and_world_models() -> None:
    prompt = student_prompt(StudentRole.WORLD_MODELER, "S-wm", context())
    assert "LATEST_TRANSITION" in prompt and "WORLD_MODELS" in prompt
    assert "HYPOTHESIS_LEDGER" not in prompt


def test_skeptic_prompt_includes_its_own_prior_reports_not_absent() -> None:
    own = (
        StudentReport(
            student_id="S-skeptic",
            role=StudentRole.SKEPTIC,
            turn=1,
            assessment="still unsure",
        ),
    )
    prompt = student_prompt(StudentRole.SKEPTIC, "S-skeptic", context(), own)
    assert "OWN_PRIOR_REPORTS" in prompt and "still unsure" in prompt


def test_peer_review_prompt_shows_peers_never_its_own_report() -> None:
    mine = StudentReport(
        student_id="S-scientist",
        role=StudentRole.SCIENTIST,
        turn=2,
        assessment="my own claim",
    )
    other = StudentReport(
        student_id="S-skeptic",
        role=StudentRole.SKEPTIC,
        turn=2,
        assessment="peer claim",
    )
    prompt = peer_review_prompt("S-scientist", StudentRole.SCIENTIST, mine, (other,), 2)
    assert "peer claim" in prompt
    assert (
        prompt.count("my own claim") == 1
    )  # only inside YOUR_PRIOR_REPORT, not PEER_REPORTS
    peer_section = prompt[prompt.index("PEER_REPORTS") :]
    assert "my own claim" not in peer_section
