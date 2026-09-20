"""One typed profile drives runtime composition and precedence."""

from pathlib import Path

import pytest

from arc_agi_3.contracts.enums import Resource
from arc_agi_3.deployment.kaggle_settings import KaggleSettings
from arc_agi_3.settings import resolve_runtime


def test_profile_environment_and_explicit_override_precedence(tmp_path: Path) -> None:
    settings = resolve_runtime(
        "colab_transformers",
        environ={"AREX_MODEL_PATH": "/weights", "AREX_MAX_MODEL_CALLS": "10"},
        overrides={"max_model_calls": 12, "output_dir": tmp_path},
    )
    assert settings.model_path == Path("/weights")
    assert settings.max_turns == 8
    assert settings.max_model_calls == 12
    assert settings.output_dir == tmp_path
    assert settings.to_budget_config().limits[Resource.MODEL_CALLS] == 12
    assert settings.to_transformers_config().max_time_seconds == 180
    assert settings.to_run_config("run", "ls20").max_turns == 8


def test_colab_model_path_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="explicit model_path"):
        resolve_runtime("colab_transformers", environ={})


def test_kaggle_profile_is_independent_and_packaged() -> None:
    settings = resolve_runtime("kaggle_submission", environ={})
    assert settings.max_turns == 40
    assert settings.max_model_calls == 40
    assert settings.max_new_tokens == 2048
    assert settings.model_path.is_absolute()


def test_cross_field_validation_and_surprising_budget_warning() -> None:
    with pytest.raises(ValueError, match="max_new_tokens exceeds"):
        resolve_runtime("local_smoke", {"output_token_budget": 5}, environ={})
    with pytest.warns(UserWarning, match="max_model_calls < max_turns"):
        resolve_runtime("local_smoke", {"max_model_calls": 1}, environ={})


def test_kaggle_from_env_uses_profile_and_metadata_model_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AREX_MODEL_PATH", "/kaggle/input/attached-model")
    monkeypatch.setenv("AREX_MAX_MODEL_CALLS", "9")
    settings = KaggleSettings.from_env()
    assert settings.model_path == Path("/kaggle/input/attached-model")
    assert settings.max_model_calls == 9
    assert settings.max_turns == 40
