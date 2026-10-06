"""Generation attempts are hard-bounded and correctly rolewise accounted for."""

import json

import pytest

from arc_agi_3.adapters.structured_model import (
    StructuredModelAdapter,
    StructuredOutputError,
)
from arc_agi_3.testing import successful_script
from tests.generation_fixtures import ScriptedBackend, context

VALID = json.dumps(successful_script()[0].model_dump(mode="json"))


def test_valid_primary_output_needs_no_repair() -> None:
    backend = ScriptedBackend([VALID])
    response = StructuredModelAdapter(backend).decide(context())
    assert [a.role for a in response.attempts] == ["primary"]
    assert response.attempts[0].valid is True
    assert len(backend.prompts) == 1


@pytest.mark.parametrize(
    "malformed,category",
    [
        ("not json at all", "no_json_found"),
        ('{"mode": "execute", "bad"', "malformed_json"),
        ('{"status": "ok"}', "schema_validation"),
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


def test_primary_plus_two_repairs_failing_terminates_cleanly() -> None:
    backend = ScriptedBackend(["x", "y", "z"])
    with pytest.raises(StructuredOutputError) as caught:
        StructuredModelAdapter(backend, max_repairs=2).decide(context())
    assert len(backend.prompts) == 3
    attempts = caught.value.trace_payload["attempts"]
    assert isinstance(attempts, list) and len(attempts) == 3
    assert [a["role"] for a in attempts] == ["primary", "repair", "repair"]


def test_no_fourth_generation_attempt_is_ever_possible() -> None:
    backend = ScriptedBackend(["x", "y", "z", "never reached"])
    with pytest.raises(StructuredOutputError):
        StructuredModelAdapter(backend, max_repairs=2).decide(context())
    assert len(backend.prompts) == 3
    with pytest.raises(ValueError, match="max_repairs"):
        StructuredModelAdapter(backend, max_repairs=3)


def test_output_hashes_are_preserved_across_attempts() -> None:
    from arc_agi_3.trace.canonical import canonical_hash

    backend = ScriptedBackend(["not json", VALID])
    response = StructuredModelAdapter(backend).decide(context())
    assert response.attempts[0].output_hash == canonical_hash("not json")
    assert response.attempts[1].output_hash == canonical_hash(VALID)


def test_structured_output_error_names_the_actual_target_type() -> None:
    backend = ScriptedBackend(["x", "y", "z"])
    with pytest.raises(StructuredOutputError, match="CognitiveDecision"):
        StructuredModelAdapter(backend, max_repairs=2).decide(context())


def test_failed_attempts_preserve_generation_token_usage() -> None:
    backend = ScriptedBackend(["not json", VALID])
    response = StructuredModelAdapter(backend).decide(context())
    assert response.attempts[0].generation_input_tokens == backend.usage.input_tokens
    assert response.attempts[0].generation_output_tokens == backend.usage.output_tokens
