"""Validate every tool_requests entry before a decision is accepted.

A malformed tool call used to surface only at execution time, as a failed
ToolResult the model would see on the *next* full cognitive turn - costing
a full-price generation per guess while it reverse-engineered the real
argument shape from runtime errors. Checking here instead means it is
repaired through the same cheap, dedicated repair path as any other
structurally invalid decision.
"""

from pydantic import ValidationError

from arc_agi_3.contracts.decision_core import CognitiveDecision
from arc_agi_3.tool_registry import TOOL_ARGUMENTS
from arc_agi_3.trace.canonical import canonical_json

from .generation_errors import ToolContractError


def check_tool_requests(decision: CognitiveDecision) -> None:
    ids = [request.request_id for request in decision.tool_requests]
    if len(ids) != len(set(ids)):
        raise ToolContractError(
            "tool_requests entries must have unique request_id values"
        )
    for request in decision.tool_requests:
        model = TOOL_ARGUMENTS.get(request.tool_name)
        if model is None:
            raise ToolContractError(f"unknown tool {request.tool_name!r}")
        try:
            model.model_validate(request.arguments)
        except ValidationError as error:
            errors = error.errors(
                include_context=False, include_input=False, include_url=False
            )
            detail = canonical_json(errors)
            raise ToolContractError(
                f"tool_requests entry for {request.tool_name!r} is invalid: {detail}"
            ) from error
