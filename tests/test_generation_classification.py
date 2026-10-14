"""Each parse/validation failure is classified distinctly and repaired."""

import json

import pytest

from arc_agi_3.adapters.structured_model import (
    StructuredModelAdapter,
    StructuredOutputError,
)
from arc_agi_3.testing import successful_script
from tests.generation_fixtures import ScriptedBackend, context

VALID = json.dumps(successful_script()[0].model_dump(mode="json"))


@pytest.mark.parametrize(
    "malformed,category",
    [
        ("not json at all", "no_json_found"),
        ('{"mode": "execute", "bad"', "malformed_json"),
        ('{"assessment": "x", "intent": "x", "mode": 123}', "schema_validation"),
        ('{"action_id": 1}', "root_type_confusion"),
    ],
)
def test_each_parse_failure_is_classified_and_repaired(
    malformed: str, category: str
) -> None:
    backend = ScriptedBackend([malformed, VALID])
    response = StructuredModelAdapter(backend).decide(context())
    assert [a.role for a in response.attempts] == ["primary", "repair"]
    assert response.attempts[0].error_category == category
    assert response.attempts[1].valid is True


def test_logical_contract_violation_is_classified_and_repaired() -> None:
    invalid = successful_script()[0].model_dump(mode="json")
    invalid["mode"] = "plan"
    backend = ScriptedBackend([json.dumps(invalid), VALID])
    response = StructuredModelAdapter(backend).decide(context())
    assert response.attempts[0].error_category == "logical_contract"
    assert response.attempts[1].valid is True


def test_root_type_confusion_and_logical_contract_are_distinct_each_time() -> None:
    action_like = json.dumps({"action_id": 1})
    invalid_mode = successful_script()[0].model_dump(mode="json")
    invalid_mode["mode"] = "plan"
    experiment_like = json.dumps(
        {"experiment_id": "e1", "question": "x", "hypothesis_ids": []}
    )
    backend = ScriptedBackend([action_like, json.dumps(invalid_mode), experiment_like])
    with pytest.raises(StructuredOutputError) as caught:
        StructuredModelAdapter(backend, max_repairs=2).decide(context())
    attempts = caught.value.trace_payload["attempts"]
    assert isinstance(attempts, list)
    assert [a["error_category"] for a in attempts] == [
        "root_type_confusion",
        "logical_contract",
        "root_type_confusion",
    ]
