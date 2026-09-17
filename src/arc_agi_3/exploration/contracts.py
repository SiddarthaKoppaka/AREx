"""Explicit semantic inputs and deterministic exploration outputs."""

from pydantic import Field

from arc_agi_3.contracts.base import Contract


class ExplorationWeights(Contract):
    information: float
    reward_progress: float
    hypothesis_discrimination: float


class ProbeEstimate(Contract):
    probe_id: str
    candidate_state_hash: str | None = None
    drig: float
    expected_reward_progress: float
    hypothesis_discrimination: float
    action_cost: float = Field(ge=0)
    inference_cost: float = Field(ge=0)
    risk_cost: float = Field(ge=0)
    repeat_cost: float = Field(ge=0)
    repeat_cost_per_visit: float = Field(default=0, ge=0)


class ExplorationRequest(Contract):
    request_id: str
    probes: tuple[ProbeEstimate, ...] = Field(min_length=1)
    weights: ExplorationWeights
    executable_plan_utility: float
    safety_margin: float = Field(default=0, ge=0)
    recent_marginal_values: tuple[float, ...] = ()
    saturation_window: int = Field(default=3, gt=0)
    saturation_threshold: float = 0.0


class ProbeScore(Contract):
    probe_id: str
    value: float
    prior_visits: int = Field(ge=0)
    novel_state: bool | None = None


class ExplorationResult(Contract):
    request_id: str
    scores: tuple[ProbeScore, ...]
    maximum_value: float
    maximizing_probe_ids: tuple[str, ...]
    exploration_dominated: bool
    saturated: bool


class ExplorationOutcome(Contract):
    state_hash: str | None = None
    marginal_value: float


class ExplorationHistorySnapshot(Contract):
    visits: dict[str, int] = Field(default_factory=dict)
    marginal_values: tuple[float, ...] = ()
