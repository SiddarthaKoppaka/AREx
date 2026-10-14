"""Classify invalid generations and give each category a repair instruction.

Classification is mechanical, never semantic: it looks only at which parser
or validator stage rejected the text, not at what the model was trying to say.
"""

import json

from pydantic import BaseModel, ValidationError

from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.contracts.generation import ErrorCategory


def has_json_candidate(text: str) -> bool:
    return "{" in text


def is_logical_contract_error(error: ValidationError) -> bool:
    """True when every failure is a custom cross-field rule, not a type/shape error."""
    errors = error.errors(include_context=False, include_input=False, include_url=False)
    return bool(errors) and all(item["type"] == "value_error" for item in errors)


def is_root_type_confusion(
    error: ValidationError, output_model: type[BaseModel]
) -> bool:
    """True when the model likely returned a nested type's shape as the root
    object: every required top-level field is missing, and there are fields
    present that the schema forbids (extra="forbid")."""
    errors = error.errors(include_context=False, include_input=False, include_url=False)
    if not any(item["type"] == "extra_forbidden" for item in errors):
        return False
    missing = {
        item["loc"][0]
        for item in errors
        if item["type"] == "missing" and len(item["loc"]) == 1
    }
    required = {
        name for name, field in output_model.model_fields.items() if field.is_required()
    }
    return bool(required) and required <= missing


class ToolContractError(ValueError):
    """A tool_requests entry does not match its registered argument contract."""


class BeliefContractError(ValueError):
    """A hypothesis mutation conflicts with the current belief state."""


def classify(
    error: Exception, text: str, output_model: type[BaseModel]
) -> ErrorCategory:
    if isinstance(error, ToolContractError):
        return "tool_contract_violation"
    if isinstance(error, BeliefContractError):
        return "belief_contract_violation"
    if isinstance(error, json.JSONDecodeError):
        return "malformed_json" if has_json_candidate(text) else "no_json_found"
    if isinstance(error, ValidationError):
        if is_root_type_confusion(error, output_model):
            return "root_type_confusion"
        if is_logical_contract_error(error):
            return "logical_contract"
        return "schema_validation"
    return "unknown_backend_failure"


class GenerationBackendError(RuntimeError):
    """A backend generation call failed before producing any output text."""

    def __init__(self, message: str, usage: ModelUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or ModelUsage()
