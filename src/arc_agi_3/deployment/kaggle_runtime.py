"""Compose one fresh AREx session for each official framework game."""

from typing import Any

from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig
from arc_agi_3.config import BudgetConfig, RunConfig
from arc_agi_3.contracts.enums import Resource
from arc_agi_3.runtime import build_session

from .callback_environment import CallbackEnvironment
from .kaggle_bridge import KaggleAgentBridge
from .kaggle_preflight import run_preflight
from .kaggle_settings import KaggleSettings

_backend: TransformersBackend | None = None


def _shared_backend(settings: KaggleSettings) -> TransformersBackend:
    global _backend
    if _backend is None:
        run_preflight(settings)
        _backend = TransformersBackend(
            TransformersConfig(
                model_path=str(settings.model_path),
                model_name=settings.model_name,
                model_digest=settings.model_digest,
                max_new_tokens=settings.max_new_tokens,
            )
        )
    return _backend


def build_kaggle_bridge(
    game_id: str,
    *,
    settings: KaggleSettings | None = None,
    backend: Any | None = None,
) -> KaggleAgentBridge:
    resolved = settings or KaggleSettings.from_env()
    inference = backend or _shared_backend(resolved)
    environment = CallbackEnvironment(game_id, framework_version="official")
    budget = BudgetConfig(
        limits={
            Resource.ACTIONS: resolved.max_actions,
            Resource.MODEL_CALLS: resolved.max_model_calls,
            Resource.INPUT_TOKENS: resolved.max_input_tokens,
            Resource.OUTPUT_TOKENS: resolved.max_output_tokens,
        }
    )
    config = RunConfig(
        run_id=f"kaggle-{game_id}",
        experiment_id="kaggle-qwen35-9b",
        game_id=game_id,
        max_turns=resolved.max_actions,
        output_dir=resolved.output_dir,
        budget=budget,
    )
    model = StructuredModelAdapter(inference, max_repairs=resolved.max_repairs)
    return KaggleAgentBridge(game_id, build_session(config, environment, model))
