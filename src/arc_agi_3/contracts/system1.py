"""Typed System-1 decision request/result contracts. See ADR-0010.

JEV (or any System1Backend) returns bounded structured judgments over a
candidate set the Core Agent itself constructed - never free-form
reasoning, never a new candidate, never semantic interpretation.
"""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .base import Contract


class CandidateOperation(Contract):
    candidate_id: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=240)
    tool_name: str | None = None
    tool_arguments: dict[str, JsonValue] | None = None
    action_id: int | None = None
    action_data: dict[str, JsonValue] | None = None


class System1DecisionRequest(Contract):
    trial_id: str = Field(min_length=1)
    goal: str = Field(min_length=1, max_length=512)
    state_summary: str = Field(min_length=1, max_length=1024)
    candidates: tuple[CandidateOperation, ...] = Field(min_length=1, max_length=8)
    execution_mode: Literal["suggest", "execute_one"] = "suggest"
    ask_stalled: bool = True
    ask_information_gain: bool = True
    ask_need_more_perception: bool = True

    @model_validator(mode="after")
    def unique_candidate_ids(self) -> "System1DecisionRequest":
        ids = [item.candidate_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        return self


class CandidateProbability(Contract):
    candidate_id: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class System1DecisionResult(Contract):
    trial_id: str = Field(min_length=1)
    selected_candidate_id: str = Field(min_length=1)
    candidate_probabilities: tuple[CandidateProbability, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    stalled_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    need_more_perception_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    information_gain_score: float | None = None
    latency_ms: int | None = Field(default=None, ge=0)
