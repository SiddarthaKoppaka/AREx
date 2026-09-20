"""Kaggle-specific entry point to the shared runtime profile."""

import os

from pydantic import model_validator

from arc_agi_3.settings import resolve_runtime
from arc_agi_3.settings_models import RuntimeSettings


class KaggleSettings(RuntimeSettings):
    profile: str = "kaggle_submission"
    require_gpu: bool = True

    @model_validator(mode="before")
    @classmethod
    def mirror_legacy_action_turns(cls, value: object) -> object:
        if (
            isinstance(value, dict)
            and "max_actions" in value
            and "max_turns" not in value
        ):
            return {**value, "max_turns": value["max_actions"]}
        return value

    @classmethod
    def from_env(cls) -> "KaggleSettings":
        if not os.getenv("AREX_MODEL_PATH"):
            raise RuntimeError("AREX_MODEL_PATH is required")
        profile = os.getenv("AREX_PROFILE", "kaggle_submission")
        return cls.model_validate(resolve_runtime(profile).model_dump())
