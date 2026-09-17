"""Explicit model-asset inventory with size and checksum verification."""

import hashlib
from pathlib import Path

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.trace.canonical import canonical_json


class AssetEntry(Contract):
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str


class ModelAssetManifest(Contract):
    model_id: str
    revision: str
    assets: tuple[AssetEntry, ...]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_asset_manifest(
    root: Path, paths: list[Path], model_id: str, revision: str
) -> ModelAssetManifest:
    resolved = root.resolve()
    entries = []
    for path in sorted(path.resolve() for path in paths):
        relative = path.relative_to(resolved)
        entries.append(
            AssetEntry(
                relative_path=str(relative),
                size_bytes=path.stat().st_size,
                sha256=_digest(path),
            )
        )
    return ModelAssetManifest(
        model_id=model_id, revision=revision, assets=tuple(entries)
    )


def verify_assets(root: Path, manifest: ModelAssetManifest) -> list[str]:
    failures = []
    for entry in manifest.assets:
        path = (root / entry.relative_path).resolve()
        if not path.is_relative_to(root.resolve()):
            failures.append(f"outside_root:{entry.relative_path}")
            continue
        if not path.is_file():
            failures.append(f"missing:{entry.relative_path}")
        elif path.stat().st_size != entry.size_bytes or _digest(path) != entry.sha256:
            failures.append(f"mismatch:{entry.relative_path}")
    return failures


def write_asset_manifest(path: Path, manifest: ModelAssetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest) + "\n")
