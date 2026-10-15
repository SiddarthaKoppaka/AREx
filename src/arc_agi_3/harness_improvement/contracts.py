"""Typed records for the external harness-improvement loop (PR-10). See
PR-10-report.md: this is measurement/bookkeeping scaffolding for a
human-driven workflow, not an autonomous patch-generator - "no automatic
production promotion" per the gate.
"""

from typing import Literal

from pydantic import Field, JsonValue

from arc_agi_3.contracts.base import Contract

PromotionStatus = Literal["proposed", "accepted", "rejected"]


class FailurePacket(Contract):
    """A machine-readable summary of one run's most salient failure
    signal, derived from an existing BaselineReport - never invented."""

    run_id: str
    game_id: str
    stop_reason: str | None
    failure_summary: str = Field(min_length=1, max_length=512)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    metrics_snapshot: dict[str, JsonValue]


class StaticCheckResult(Contract):
    check: str
    passed: bool
    output_tail: str = Field(default="", max_length=4000)


class HarnessChangeCandidate(Contract):
    candidate_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    diff_sha256: str = Field(min_length=64, max_length=64)
    scope_description: str = Field(min_length=1, max_length=512)
    failure_packet_refs: tuple[str, ...] = Field(min_length=1)
    static_checks: tuple[StaticCheckResult, ...] = ()
    promotion_status: PromotionStatus = "proposed"


class PromotionDecision(Contract):
    candidate_id: str
    metric: str
    accepted: bool
    reason: str = Field(min_length=1, max_length=512)
    control_value: float | int | None
    candidate_value: float | int | None
