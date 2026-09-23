"""Resolve typed runtime settings with explicit, documented precedence."""

import os
import tomllib
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .settings_limits import normalize_context_limits
from .settings_models import RuntimeSettings

REPOSITORY_CONFIG = Path(__file__).resolve().parents[2] / "configs/runtime.toml"
PACKAGE_CONFIG = Path(__file__).resolve().parent / "configs/runtime.toml"
DEFAULT_CONFIG = REPOSITORY_CONFIG if REPOSITORY_CONFIG.is_file() else PACKAGE_CONFIG


def resolve_runtime(
    profile: str,
    overrides: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> RuntimeSettings:
    """Apply typed defaults, profile, AREX environment, then explicit overrides."""
    location = config_path or DEFAULT_CONFIG
    profiles: dict[str, Any] = {}
    if location.is_file():
        with location.open("rb") as stream:
            profiles = tomllib.load(stream).get("profiles", {})
    if profile not in profiles and location.is_file():
        raise ValueError(f"unknown runtime profile: {profile}")
    defaults = RuntimeSettings().model_dump()
    values: dict[str, object] = {
        **defaults,
        "profile": profile,
        **profiles.get(profile, {}),
    }
    source = os.environ if environ is None else environ
    if (
        profile == "colab_transformers"
        and "AREX_MODEL_PATH" not in source
        and "model_path" not in (overrides or {})
    ):
        raise ValueError("colab_transformers requires an explicit model_path override")
    for name in RuntimeSettings.model_fields:
        key = f"AREX_{name.upper()}"
        if key in source:
            values[name] = source[key]
    if (
        "AREX_HARD_INPUT_LIMIT" in source
        and "AREX_MAX_INPUT_TOKENS" in source
        and source["AREX_HARD_INPUT_LIMIT"] != source["AREX_MAX_INPUT_TOKENS"]
    ):
        raise ValueError("conflicting hard and legacy input limits")
    if "AREX_HARD_INPUT_LIMIT" in source and "AREX_MAX_INPUT_TOKENS" not in source:
        values["max_input_tokens"] = source["AREX_HARD_INPUT_LIMIT"]
    elif "AREX_MAX_INPUT_TOKENS" in source and "AREX_HARD_INPUT_LIMIT" not in source:
        values["hard_input_limit"] = source["AREX_MAX_INPUT_TOKENS"]
    values.update(overrides or {})
    if (
        overrides
        and "hard_input_limit" in overrides
        and "max_input_tokens" in overrides
        and overrides["hard_input_limit"] != overrides["max_input_tokens"]
    ):
        raise ValueError("conflicting hard and legacy input limits")
    if (
        overrides
        and "hard_input_limit" in overrides
        and "max_input_tokens" not in overrides
    ):
        values["max_input_tokens"] = overrides["hard_input_limit"]
    elif (
        overrides
        and "max_input_tokens" in overrides
        and "hard_input_limit" not in overrides
    ):
        values["hard_input_limit"] = overrides["max_input_tokens"]
    normalize_context_limits(values, source, overrides or {})
    settings = RuntimeSettings.model_validate(values)
    if settings.max_model_calls < settings.max_turns:
        warnings.warn(
            "max_model_calls < max_turns; one decision per turn may stop early",
            stacklevel=2,
        )
    return settings
