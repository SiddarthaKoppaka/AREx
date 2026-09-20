"""Persistent model staging checks local disk before copying."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_agi_3.research.model_staging import stage_model


def test_stage_model_copies_once_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "drive" / "weights"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"weights")
    cache = tmp_path / "content"
    staged = stage_model(source, cache, enabled=True)
    assert staged != source
    assert (staged / "model.safetensors").read_bytes() == b"weights"
    assert (source / "model.safetensors").read_bytes() == b"weights"
    assert stage_model(source, cache, enabled=True) == staged
    (source / "model.safetensors").write_bytes(b"new-weights")
    assert stage_model(source, cache, enabled=True) != staged


def test_stage_model_checks_disk_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "weights"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        "arc_agi_3.research.model_staging.shutil.disk_usage",
        lambda path: SimpleNamespace(free=1),
    )
    with pytest.raises(OSError, match="model staging needs"):
        stage_model(source, tmp_path / "content", enabled=True)
    assert stage_model(source, tmp_path / "unused", enabled=False) == source
