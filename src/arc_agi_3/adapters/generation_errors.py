"""Classify invalid generations and give each category a repair instruction.

Classification is mechanical, never semantic: it looks only at which parser
or validator stage rejected the text, not at what the model was trying to say.
"""

import json

from pydantic import ValidationError

from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.contracts.generation import ErrorCategory

CATEGORY_INSTRUCTIONS: dict[ErrorCategory, str] = {
    "no_json_found": (
        "Return ONLY one valid JSON object matching CognitiveDecision. Do not "
        "include analysis, markdown, or prose."
    ),
    "malformed_json": (
        "A JSON object was started but did not parse. Close every brace/quote "
        "and remove trailing commas or comments."
    ),
    "schema_validation": (
        "One or more fields have the wrong type, are missing, or are not "
        "permitted by the schema. Repair only the structured object."
    ),
    "logical_contract": (
        "The object is valid JSON but violates a cross-field rule (see "
        "ERROR_DETAIL). Adjust only the fields the rule names."
    ),
    "unknown_backend_failure": (
        "The previous generation failed before producing output. Return a "
        "minimal valid CognitiveDecision consistent with your prior intent."
    ),
}


def has_json_candidate(text: str) -> bool:
    return "{" in text


def is_logical_contract_error(error: ValidationError) -> bool:
    """True when every failure is a custom cross-field rule, not a type/shape error."""
    errors = error.errors(include_context=False, include_input=False, include_url=False)
    return bool(errors) and all(item["type"] == "value_error" for item in errors)


def classify(error: Exception, text: str) -> ErrorCategory:
    if isinstance(error, json.JSONDecodeError):
        return "malformed_json" if has_json_candidate(text) else "no_json_found"
    if isinstance(error, ValidationError):
        if is_logical_contract_error(error):
            return "logical_contract"
        return "schema_validation"
    return "unknown_backend_failure"


class GenerationBackendError(RuntimeError):
    """A backend generation call failed before producing any output text."""

    def __init__(self, message: str, usage: ModelUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or ModelUsage()
