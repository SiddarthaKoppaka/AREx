"""Provider-neutral planning request and result contracts."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.enums import SearchMethod
from arc_agi_3.contracts.observation import Action
from arc_agi_3.world_model.contracts import SymbolicState


class SearchBudget(Contract):
    max_node_expansions: int = Field(gt=0, le=1_000_000)
    max_depth: int = Field(gt=0, le=10_000)


class SearchRequest(Contract):
    search_id: str
    model_id: str
    model_version: int = Field(ge=1)
    method: SearchMethod
    objective: str = Field(min_length=1)
    start_state: SymbolicState
    goal_facts: dict[str, JsonValue]
    heuristic_weights: dict[str, float] = Field(default_factory=dict)
    budget: SearchBudget

    @model_validator(mode="after")
    def valid_heuristic(self) -> "SearchRequest":
        if any(value < 0 for value in self.heuristic_weights.values()):
            raise ValueError("heuristic weights must be non-negative")
        unknown = set(self.heuristic_weights).difference(self.goal_facts)
        if unknown:
            raise ValueError("heuristic weights may name only goal facts")
        return self


type SearchStatus = Literal["found", "partial", "exhausted"]


class SearchResult(Contract):
    search_id: str
    method: SearchMethod
    status: SearchStatus
    actions: tuple[Action, ...] = ()
    states: tuple[SymbolicState, ...] = ()
    total_cost: float = Field(default=0, ge=0)
    node_expansions: int = Field(ge=0)
    reached_depth: int = Field(ge=0)
    frontier_state_hashes: tuple[str, ...] = ()
