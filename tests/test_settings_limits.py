"""Pressure, target, and emergency ceiling remain ordered."""

import pytest

from arc_agi_3.settings import resolve_runtime


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"soft_input_limit": 7680}, "soft_input_limit"),
        ({"hard_input_limit": 8192}, "hard_input_limit"),
        ({"hard_input_limit": 7700}, "generation room"),
    ],
)
def test_context_limit_ordering(overrides: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_runtime("local_smoke", overrides, environ={})


def test_legacy_and_hard_input_limits_stay_synchronized() -> None:
    legacy = resolve_runtime("local_smoke", environ={"AREX_MAX_INPUT_TOKENS": "7000"})
    explicit = resolve_runtime("local_smoke", {"hard_input_limit": 7000}, environ={})
    assert legacy.hard_input_limit == legacy.max_input_tokens == 7000
    assert explicit.hard_input_limit == explicit.max_input_tokens == 7000
    assert legacy.template_and_generation_margin == 680
    assert explicit.template_and_generation_margin == 680


def test_pressure_and_margin_recompute_emergency_ceiling() -> None:
    settings = resolve_runtime(
        "kaggle_submission",
        {"compaction_pressure_start": 10000, "template_and_generation_margin": 768},
        environ={},
    )
    assert settings.soft_input_limit == 10000
    assert settings.hard_input_limit == settings.max_input_tokens == 29952
    assert settings.to_transformers_config().max_input_tokens == 29952


def test_active_target_cannot_exceed_model_headroom() -> None:
    with pytest.raises(ValueError, match="active context target"):
        resolve_runtime(
            "colab_transformers",
            {"model_path": "/weights", "active_context_target": 16000},
            environ={},
        )


def test_conflicting_legacy_and_hard_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        resolve_runtime(
            "local_smoke",
            {"hard_input_limit": 7000, "max_input_tokens": 7100},
            environ={},
        )
