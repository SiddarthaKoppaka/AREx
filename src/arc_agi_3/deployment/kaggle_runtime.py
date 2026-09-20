"""Compose one fresh AREx session for each official framework game."""

from typing import Any

from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig
from arc_agi_3.runtime import build_session

from .callback_environment import CallbackEnvironment
from .kaggle_bridge import KaggleAgentBridge
from .kaggle_preflight import run_preflight
from .kaggle_settings import KaggleSettings

_backend: TransformersBackend | None = None
_backend_config: TransformersConfig | None = None


def _shared_backend(settings: KaggleSettings) -> TransformersBackend:
    global _backend, _backend_config
    config = settings.to_transformers_config()
    if _backend is None or _backend_config != config:
        run_preflight(settings)
        _backend = TransformersBackend(config)
        _backend_config = config
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
    config = resolved.to_run_config(
        run_id=f"kaggle-{game_id}",
        game_id=game_id,
        experiment_id="kaggle-qwen3-8b",
    )
    model = resolved.to_structured_adapter(inference)
    return KaggleAgentBridge(game_id, build_session(config, environment, model))
