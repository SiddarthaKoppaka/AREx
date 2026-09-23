"""Compact, evidence-linked working memory controlled through typed operations."""

from pydantic import Field

from .base import Contract


class VerifiedFact(Contract):
    fact_id: str
    version: int = Field(ge=1)
    fact: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    supersedes: tuple[str, ...] = ()


class NextTest(Contract):
    action_id: int = Field(ge=0, le=7)
    purpose: str


class WorkingScratchpad(Contract):
    version: int = Field(default=1, ge=1)
    objective: str = "Understand the game and make progress."
    verified_facts: tuple[VerifiedFact, ...] = ()
    action_model: dict[str, str] = Field(default_factory=dict)
    active_plan: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    last_useful_result: str | None = None
    next_test: NextTest | None = None
    capabilities: dict[str, bool] = Field(default_factory=dict)


class ScratchpadUpdates(Contract):
    set_objective: str | None = None
    add_verified_fact: tuple[VerifiedFact, ...] = ()
    action_model_updates: dict[str, str] = Field(default_factory=dict)
    revise_plan: tuple[str, ...] | None = None
    add_open_questions: tuple[str, ...] = ()
    resolve_open_questions: tuple[str, ...] = ()
    set_last_useful_result: str | None = None
    set_next_test: NextTest | None = None
