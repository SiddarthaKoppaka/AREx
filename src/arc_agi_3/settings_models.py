"""Typed values shared by local, Colab, and Kaggle runtime profiles."""

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import model_validator

from .settings_fields import RuntimeFields
from .settings_validation import validate_runtime

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arc_agi_3.adapters.inference import InferenceBackend
    from arc_agi_3.adapters.protocols import EnvironmentAdapter
    from arc_agi_3.adapters.structured_model import StructuredModelAdapter
    from arc_agi_3.adapters.system1 import System1Backend
    from arc_agi_3.adapters.transformers import TransformersConfig
    from arc_agi_3.config import BudgetConfig, RunConfig
    from arc_agi_3.runtime.reporting import LiveReporter
    from arc_agi_3.runtime.runner import EpisodeRunner


class RuntimeSettings(RuntimeFields):
    @model_validator(mode="after")
    def validate_budgets(self) -> "RuntimeSettings":
        return validate_runtime(self)

    @classmethod
    def load(
        cls,
        profile: str,
        overrides: "Mapping[str, object] | None" = None,
        environ: "Mapping[str, str] | None" = None,
        config_path: Path | None = None,
    ) -> "RuntimeSettings":
        from .settings import resolve_runtime

        return cls.model_validate(
            resolve_runtime(profile, overrides, environ, config_path).model_dump()
        )

    def to_transformers_config(
        self, model_path: str | Path | None = None
    ) -> "TransformersConfig":
        from .settings_composition import transformers_config

        return transformers_config(self, model_path)

    def to_budget_config(self) -> "BudgetConfig":
        from .settings_composition import budget_config

        return budget_config(self)

    def to_run_config(
        self,
        run_id: str,
        game_id: str,
        seed: int = 0,
        experiment_id: str | None = None,
    ) -> "RunConfig":
        from .settings_composition import run_config

        return run_config(self, run_id, game_id, seed, experiment_id)

    def to_structured_adapter(
        self, backend: "InferenceBackend"
    ) -> "StructuredModelAdapter":
        from .settings_composition import structured_adapter

        return structured_adapter(self, backend)

    def build_runner(
        self,
        run_id: str,
        game_id: str,
        environment: "EnvironmentAdapter",
        backend: "InferenceBackend",
        seed: int = 0,
        reporter: "LiveReporter | None" = None,
        raise_on_failure: bool = False,
        *,
        specialist_backend: "InferenceBackend | None" = None,
        system1_backend: "System1Backend | None" = None,
    ) -> "EpisodeRunner":
        from arc_agi_3.runtime import build_runner

        config = self.to_run_config(run_id, game_id, seed)
        return build_runner(
            config,
            environment,
            self.to_structured_adapter(backend),
            specialist_backend=specialist_backend,
            system1_backend=system1_backend,
            reporter=reporter,
            raise_on_failure=raise_on_failure,
        )
