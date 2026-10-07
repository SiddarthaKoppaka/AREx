"""Typed roles and failure categories for one backend text generation."""

from typing import Literal

GenerationRole = Literal["primary", "repair"]

ErrorCategory = Literal[
    "no_json_found",
    "malformed_json",
    "schema_validation",
    "logical_contract",
    "tool_contract_violation",
    "belief_contract_violation",
    "unknown_backend_failure",
]
