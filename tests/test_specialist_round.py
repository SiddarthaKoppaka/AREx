"""Specialist report generation reuses the one bounded generation engine,
independently from context."""

import json

from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.enums import CognitiveFaculty
from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.runtime.specialist_round import generate_specialist_report
from arc_agi_3.testing import FakeLineEnvironment
from tests.generation_fixtures import ScriptedBackend

VALID_REPORT = json.dumps(
    {
        "specialist_id": "S-hypothesis",
        "faculty": "hypothesis",
        "turn": 1,
        "assessment": "ok",
    }
)


def context() -> AgentContext:
    return AgentContext(turn=1, observation=FakeLineEnvironment().reset(), budget={})


def test_malformed_specialist_output_is_repaired_the_same_cheap_way() -> None:
    backend = ScriptedBackend(["not json", VALID_REPORT])
    report, usage, attempts = generate_specialist_report(
        backend,
        CognitiveFaculty.HYPOTHESIS,
        "S-hypothesis",
        context(),
        [],
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
        own_history_limit=2,
    )
    assert isinstance(report, SpecialistReport)
    assert [a.role for a in attempts] == ["primary", "repair"]
    assert attempts[0].error_category == "no_json_found"
    assert "CURRENT_OBSERVATION" not in backend.prompts[1]
    assert usage.input_tokens == 4
    assert (
        "SpecialistReport" in backend.prompts[1]
        and "CognitiveDecision" not in (backend.prompts[1])
    )
    assert attempts[0].generation_input_tokens == backend.usage.input_tokens


def test_two_specialists_from_the_same_context_never_see_each_other() -> None:
    shared = context()
    first = ScriptedBackend([VALID_REPORT])
    second = ScriptedBackend(
        [
            json.dumps(
                {
                    "specialist_id": "S-dynamics",
                    "faculty": "dynamics",
                    "turn": 1,
                    "assessment": "ok",
                }
            )
        ]
    )
    generate_specialist_report(
        first,
        CognitiveFaculty.HYPOTHESIS,
        "S-hypothesis",
        shared,
        [],
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
        own_history_limit=2,
    )
    generate_specialist_report(
        second,
        CognitiveFaculty.DYNAMICS,
        "S-dynamics",
        shared,
        [],
        max_repairs=2,
        persist_invalid_output=False,
        preview_chars=300,
        own_history_limit=2,
    )
    assert "S-hypothesis" not in second.prompts[0]
    assert "S-dynamics" not in first.prompts[0]
