"""Fail-fast checks for the attached Kaggle inference runtime."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_agi_3.deployment import KaggleSettings, run_preflight


def _settings(tmp_path: Path) -> KaggleSettings:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    return KaggleSettings(
        model_path=tmp_path,
        output_dir=tmp_path / "output",
        require_gpu=False,
    )


def test_preflight_accepts_complete_model_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        "arc_agi_3.deployment.kaggle_preflight.import_module",
        lambda name: torch if name == "torch" else object(),
    )
    assert run_preflight(_settings(tmp_path))["cuda"] is False


def test_preflight_lists_missing_assets_and_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        "arc_agi_3.deployment.kaggle_preflight.import_module",
        lambda name: torch if name == "torch" else object(),
    )
    settings = KaggleSettings(model_path=tmp_path, output_dir=tmp_path / "output")
    with pytest.raises(RuntimeError, match=r"config.json.*safetensors.*CUDA"):
        run_preflight(settings)
