"""Classify invalid generations and give each category a repair instruction.

Classification is mechanical, never semantic: it looks only at which parser
or validator stage rejected the text, not at what the model was trying to say.
"""

import json

from pydantic import ValidationError

from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.contracts.generation import ErrorCategory


def has_json_candidate(text: str) -> bool:
    return "{" in text


def is_logical_contract_error(error: ValidationError) -> bool:
    """True when every failure is a custom cross-field rule, not a type/shape error."""
    errors = error.errors(include_context=False, include_input=False, include_url=False)
    return bool(errors) and all(item["type"] == "value_error" for item in errors)


class ToolContractError(ValueError):
    """A tool_requests entry does not match its registered argument contract."""


class BeliefContractError(ValueError):
    """A hypothesis mutation conflicts with the current belief state."""


class ScratchpadContractError(ValueError):
    """A scratchpad update violates working-memory rules or its token budget."""


def classify(error: Exception, text: str) -> ErrorCategory:
    if isinstance(error, ToolContractError):
        return "tool_contract_violation"
    if isinstance(error, BeliefContractError):
        return "belief_contract_violation"
    if isinstance(error, ScratchpadContractError):
        return "scratchpad_contract_violation"
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
