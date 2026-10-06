"""Students reuse the one bounded generation engine, independently from context."""

import json

from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import StudentRole
from arc_agi_3.runtime.classroom_round import (
    generate_peer_review,
    generate_student_report,
)
from arc_agi_3.testing import FakeLineEnvironment
from tests.generation_fixtures import ScriptedBackend

VALID_REPORT = json.dumps(
    {"student_id": "S-scientist", "role": "scientist", "turn": 1, "assessment": "ok"}
)


def context() -> AgentContext:
    return AgentContext(turn=1, observation=FakeLineEnvironment().reset(), budget={})


def test_malformed_student_output_is_repaired_the_same_cheap_way() -> None:
    backend = ScriptedBackend(["not json", VALID_REPORT])
    report, usage, attempts = generate_student_report(
        backend,
        StudentRole.SCIENTIST,
        "S-scientist",
        context(),
        [],
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
        own_history_limit=2,
    )
    assert isinstance(report, StudentReport)
    assert [a.role for a in attempts] == ["primary", "repair"]
    assert attempts[0].error_category == "no_json_found"
    assert "CURRENT_OBSERVATION" not in backend.prompts[1]
    assert usage.input_tokens == 4
    assert "StudentReport" in backend.prompts[1] and "CognitiveDecision" not in (
        backend.prompts[1]
    )
    assert attempts[0].generation_input_tokens == backend.usage.input_tokens


def test_two_students_from_the_same_context_never_see_each_other() -> None:
    shared = context()
    first = ScriptedBackend([VALID_REPORT])
    second = ScriptedBackend(
        [
            json.dumps(
                {
                    "student_id": "S-wm",
                    "role": "world_modeler",
                    "turn": 1,
                    "assessment": "ok",
                }
            )
        ]
    )
    generate_student_report(
        first,
        StudentRole.SCIENTIST,
        "S-scientist",
        shared,
        [],
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
        own_history_limit=2,
    )
    generate_student_report(
        second,
        StudentRole.WORLD_MODELER,
        "S-wm",
        shared,
        [],
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
        own_history_limit=2,
    )
    assert "S-scientist" not in second.prompts[0]
    assert "S-wm" not in first.prompts[0]


def test_peer_review_reuses_the_same_engine_and_sees_only_peers() -> None:
    mine = StudentReport(
        student_id="S-scientist", role=StudentRole.SCIENTIST, turn=1, assessment="mine"
    )
    peer = StudentReport(
        student_id="S-skeptic", role=StudentRole.SKEPTIC, turn=1, assessment="peer"
    )
    valid_review = json.dumps(
        {"student_id": "S-scientist", "role": "scientist", "turn": 1}
    )
    backend = ScriptedBackend(["bad", valid_review])
    review, _, attempts = generate_peer_review(
        backend,
        mine,
        (peer,),
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
    )
    assert review.student_id == "S-scientist"
    assert [a.role for a in attempts] == ["primary", "repair"]
    assert (
        "peer" in backend.prompts[0]
        and "mine" not in backend.prompts[0].split("PEER_REPORTS")[1]
    )
    assert "PeerReview" in backend.prompts[1] and "CognitiveDecision" not in (
        backend.prompts[1]
    )
