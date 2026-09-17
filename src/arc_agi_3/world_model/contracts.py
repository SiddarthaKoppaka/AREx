"""Versioned declarative transition-model contracts."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.observation import Action
from arc_agi_3.trace.canonical import canonical_hash


class SymbolicState(Contract):
    facts: dict[str, JsonValue]
    state_hash: str

    @classmethod
    def build(cls, facts: dict[str, JsonValue]) -> "SymbolicState":
        return cls(facts=facts, state_hash=canonical_hash(facts))

    @model_validator(mode="after")
    def valid_hash(self) -> "SymbolicState":
        if self.state_hash != canonical_hash(self.facts):
            raise ValueError("symbolic state hash does not match facts")
        return self


class DeclarativeRule(Contract):
    rule_id: str
    action: Action
    preconditions: dict[str, JsonValue] = Field(default_factory=dict)
    effects: dict[str, JsonValue] = Field(default_factory=dict)
    cost: float = Field(default=1.0, gt=0)


class WorldModel(Contract):
    model_id: str
    version: int = Field(ge=1)
    rules: tuple[DeclarativeRule, ...]
    assumptions: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_rules(self) -> "WorldModel":
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("world-model rule IDs must be unique")
        return self


class SimulationRequest(Contract):
    model_id: str
    model_version: int = Field(ge=1)
    initial_state: SymbolicState
    actions: tuple[Action, ...] = Field(min_length=1, max_length=256)


class SimulationStep(Contract):
    index: int = Field(ge=0)
    action: Action
    before: SymbolicState
    after: SymbolicState | None = None
    rule_id: str | None = None
    status: Literal["applied", "invalid", "ambiguous"]


class SimulationResult(Contract):
    model_id: str
    model_version: int
    status: Literal["complete", "partial", "failed"]
    steps: tuple[SimulationStep, ...]
    final_state: SymbolicState


class ReplayCase(Contract):
    case_id: str
    before: SymbolicState
    action: Action
    expected_after: SymbolicState


class ReplayCheck(Contract):
    case_id: str
    passed: bool
    predicted_hash: str | None
    expected_hash: str
