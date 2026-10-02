"""Compact, evidence-linked working memory controlled through typed operations.

Planning vocabulary: `next_test` is the model's *planned* experiment. It is
epistemic memory, never executor authority; only a decision's `action` or
`action_chunk` is a requested action, and only ACTION events are executed.
Interpretations without environment evidence belong in hypotheses, not facts.
"""

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
    experiment_id: str | None = None
    hypothesis_ids: tuple[str, ...] = ()


class PlanRevision(Contract):
    """Record of the model replacing or clearing its planned experiment."""

    scratchpad_version: int = Field(ge=1)
    previous: NextTest | None
    revised: NextTest | None
    reason: str | None = None


class WorkingScratchpad(Contract):
    version: int = Field(default=1, ge=1)
    objective: str = "Understand the game and make progress."
    verified_facts: tuple[VerifiedFact, ...] = ()
    action_model: dict[str, str] = Field(default_factory=dict)
    active_plan: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    last_useful_result: str | None = None
    next_test: NextTest | None = None
    plan_revisions: tuple[PlanRevision, ...] = Field(default=(), max_length=4)
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
    clear_next_test: bool = False
    next_test_revision_reason: str | None = Field(default=None, max_length=240)
