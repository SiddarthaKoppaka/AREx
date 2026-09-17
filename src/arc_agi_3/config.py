"""Typed, hashable runtime and ablation configuration."""

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.enums import Resource
from arc_agi_3.trace.canonical import canonical_hash


class BudgetConfig(Contract):
    limits: dict[Resource, int] = Field(
        default_factory=lambda: {
            Resource.ACTIONS: 16,
            Resource.MODEL_CALLS: 16,
            Resource.INPUT_TOKENS: 1_000_000,
            Resource.OUTPUT_TOKENS: 1_000_000,
        }
    )


class EvaluatorConfig(Contract):
    level_score_cap: float = Field(default=1.15, gt=0)
    human_action_baselines: tuple[int, ...] = ()


class BeliefConfig(Contract):
    support: dict[str, float] = Field(
        default_factory=lambda: {"weak": 1.25, "moderate": 2.0, "strong": 4.0}
    )
    weaken: dict[str, float] = Field(
        default_factory=lambda: {"weak": 0.8, "moderate": 0.5, "strong": 0.25}
    )
    contradict: dict[str, float] = Field(
        default_factory=lambda: {"weak": 0.5, "moderate": 0.25, "strong": 0.1}
    )

    def multiplier(self, operation: str, strength: str) -> float:
        values = {
            "support": self.support,
            "weaken": self.weaken,
            "contradict": self.contradict,
        }.get(operation)
        if values is None or strength not in values:
            raise ValueError(f"no multiplier for {operation}/{strength}")
        value = values[strength]
        if value <= 0:
            raise ValueError("belief multipliers must be positive")
        return value


class AblationConfig(Contract):
    persistent_memory: bool = True
    hypotheses: bool = True
    tasks: bool = True
    search: bool = True
    world_models: bool = True
    prediction_verification: bool = True
    selective_retrieval: bool = True
    multi_candidate_inference: bool = True


class RunConfig(Contract):
    run_id: str
    experiment_id: str
    episode_id: str = "episode-0"
    branch_id: str = "main"
    game_id: str = "fake-line"
    seed: int = 0
    max_turns: int = Field(default=16, gt=0)
    output_dir: Path = Field(default=Path("runs"), exclude=True)
    checkpoint_every: int = Field(default=1, gt=0)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    belief: BeliefConfig = Field(default_factory=BeliefConfig)
    ablations: AblationConfig = Field(default_factory=AblationConfig)

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"output_dir"}))


def load_config(path: Path) -> RunConfig:
    if path.suffix == ".toml":
        with path.open("rb") as stream:
            values: dict[str, Any] = tomllib.load(stream)
    elif path.suffix == ".json":
        values = json.loads(path.read_text())
    else:
        raise ValueError("configuration must be TOML or JSON")
    return RunConfig.model_validate(values)
