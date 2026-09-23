"""Versionable LM-authored cognitive records."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .base import Contract
from .enums import BeliefOperation, Resource, TaskStatus
from .observation import Action


class Hypothesis(Contract):
    hypothesis_id: str
    version: int = Field(ge=1)
    claim: str
    probability: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...] = ()
    belief_group: str | None = None
    status: Literal["active", "suspended", "rejected"] = "active"


class BeliefUpdate(Contract):
    hypothesis_id: str
    expected_version: int = Field(ge=1)
    operation: BeliefOperation
    strength: Literal["weak", "moderate", "strong"]
    evidence_refs: tuple[str, ...]
    revised_claim: str | None = None
    related_hypothesis_ids: tuple[str, ...] = ()


class TaskHandoff(Contract):
    summary: str = Field(min_length=1, max_length=1024)
    artifact_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    unresolved_questions: tuple[str, ...] = ()


class TaskRecord(Contract):
    task_id: str
    version: int = Field(ge=1)
    purpose: str = Field(min_length=1, max_length=240)
    success_criteria: str = Field(min_length=1, max_length=240)
    dependencies: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    hypothesis_refs: tuple[str, ...] = ()
    budget_limits: dict[Resource, int] = Field(default_factory=dict)
    priority: int = 0
    status: TaskStatus = TaskStatus.OPEN
    handoff: TaskHandoff | None = None

    @model_validator(mode="after")
    def valid_budget(self) -> "TaskRecord":
        if any(amount < 0 for amount in self.budget_limits.values()):
            raise ValueError("task budget limits must be non-negative")
        if self.task_id in self.dependencies:
            raise ValueError("a task cannot depend on itself")
        return self


class TaskView(Contract):
    task: TaskRecord
    structurally_blocked: bool
    stale: bool


class PlanRecord(Contract):
    plan_id: str
    objective: str
    actions: tuple[Action, ...] = ()
    assumptions: tuple[str, ...] = ()


class ToolRequest(Contract):
    request_id: str
    tool_name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResult(Contract):
    request_id: str
    status: Literal["complete", "partial", "failed"]
    output: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
