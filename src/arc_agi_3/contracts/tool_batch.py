"""Typed summary of one decision's tool_requests batch. See PR-07-report.md."""

from pydantic import Field

from .base import Contract


class ToolBatchSummary(Contract):
    decision_event_id: str
    requested: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    request_ids_in_order: tuple[str, ...] = ()
