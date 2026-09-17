"""Structured public decision interface for the LM agent."""

from pydantic import Field, model_validator

from arc_agi_3.world_model.contracts import WorldModel

from .base import Contract
from .cognition import (
    BeliefUpdate,
    Hypothesis,
    PlanRecord,
    TaskRecord,
    TaskView,
    ToolRequest,
)
from .enums import DecisionMode, Resource
from .execution import ActionChunk
from .execution import ExpectedOutcome as ExpectedOutcome
from .observation import Action, Observation
from .recovery import RecoveryEvidence, RecoveryRequest
from .retrieval import ContextEvent


class BudgetRequest(Contract):
    resource: Resource
    amount: int = Field(gt=0)


class CognitiveDecision(Contract):
    assessment: str
    intent: str
    mode: DecisionMode
    evidence_refs: tuple[str, ...] = ()
    hypothesis_proposals: tuple[Hypothesis, ...] = ()
    hypothesis_updates: tuple[BeliefUpdate, ...] = ()
    task_updates: tuple[TaskRecord, ...] = ()
    world_model_updates: tuple[WorldModel, ...] = ()
    plan: PlanRecord | None = None
    tool_requests: tuple[ToolRequest, ...] = ()
    action: Action | None = None
    action_chunk: ActionChunk | None = None
    expected_outcome: ExpectedOutcome | None = None
    budget_requests: tuple[BudgetRequest, ...] = ()
    uncertainty: str | None = None
    recovery: RecoveryRequest | None = None

    @model_validator(mode="after")
    def action_matches_mode(self) -> "CognitiveDecision":
        authorizations = int(self.action is not None) + int(
            self.action_chunk is not None
        )
        if self.mode is DecisionMode.EXECUTE and authorizations != 1:
            raise ValueError("execute decisions require exactly one LM authorization")
        if self.mode is not DecisionMode.EXECUTE and authorizations:
            raise ValueError("only execute decisions may authorize actions")
        if (self.mode is DecisionMode.RECOVER) != (self.recovery is not None):
            raise ValueError("recover mode requires exactly one recovery request")
        return self


class AgentContext(Contract):
    turn: int = Field(ge=0)
    observation: Observation
    budget: dict[Resource, int]
    recent_event_refs: tuple[str, ...] = ()
    recent_events: tuple[ContextEvent, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    tasks: tuple[TaskView, ...] = ()
    world_models: tuple[WorldModel, ...] = ()
    recovery_evidence: RecoveryEvidence | None = None


class ModelUsage(Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class ModelAttempt(Contract):
    attempt: int = Field(ge=1)
    output_hash: str
    valid: bool
    validation_error: str | None = None


class ModelResponse(Contract):
    decision: CognitiveDecision
    usage: ModelUsage = Field(default_factory=ModelUsage)
    attempts: tuple[ModelAttempt, ...] = ()
