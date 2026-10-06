"""Specialist review generation reuses the one bounded generation engine
and sees only other faculties' reports, never its own."""

import json

from arc_agi_3.contracts.enums import CognitiveFaculty
from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.runtime.specialist_round import generate_specialist_review
from tests.generation_fixtures import ScriptedBackend


def test_specialist_review_reuses_the_same_engine_and_sees_only_others() -> None:
    mine = SpecialistReport(
        specialist_id="S-hypothesis",
        faculty=CognitiveFaculty.HYPOTHESIS,
        turn=1,
        assessment="mine",
    )
    other = SpecialistReport(
        specialist_id="S-critic",
        faculty=CognitiveFaculty.CRITIC,
        turn=1,
        assessment="other",
    )
    valid_review = json.dumps(
        {"specialist_id": "S-hypothesis", "faculty": "hypothesis", "turn": 1}
    )
    backend = ScriptedBackend(["bad", valid_review])
    review, _, attempts = generate_specialist_review(
        backend,
        mine,
        (other,),
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
    )
    assert review.specialist_id == "S-hypothesis"
    assert [a.role for a in attempts] == ["primary", "repair"]
    assert (
        "other" in backend.prompts[0]
        and "mine" not in backend.prompts[0].split("OTHER_REPORTS")[1]
    )
    assert (
        "SpecialistReview" in backend.prompts[1]
        and "CognitiveDecision" not in (backend.prompts[1])
    )
