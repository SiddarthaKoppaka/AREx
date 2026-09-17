"""Offline deployment inventory tests."""

from pathlib import Path

from arc_agi_3.deployment import (
    AssetEntry,
    ModelAssetManifest,
    build_asset_manifest,
    detect_hardware,
    verify_assets,
)
from arc_agi_3.deployment.kaggle import offline_smoke


def test_asset_manifest_detects_mutation(tmp_path: Path) -> None:
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"model-weights")
    manifest = build_asset_manifest(tmp_path, [asset], "fake", "revision-1")
    assert verify_assets(tmp_path, manifest) == []
    asset.write_bytes(b"changed")
    assert verify_assets(tmp_path, manifest) == ["mismatch:weights.bin"]


def test_hardware_profile_is_serializable() -> None:
    profile = detect_hardware()
    assert profile.cpu_count >= 1
    assert profile.model_dump(mode="json")["system"]


def test_asset_verification_rejects_paths_outside_root(tmp_path: Path) -> None:
    manifest = ModelAssetManifest(
        model_id="fake",
        revision="one",
        assets=(AssetEntry(relative_path="../secret", size_bytes=0, sha256="x"),),
    )
    assert verify_assets(tmp_path, manifest) == ["outside_root:../secret"]


def test_thin_offline_entrypoint_runs(tmp_path: Path) -> None:
    assert offline_smoke(tmp_path).succeeded
