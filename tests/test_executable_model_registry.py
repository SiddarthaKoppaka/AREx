"""The candidate registry enforces parent lineage and lets competing
model_ids coexist, mirroring world_model/store.py's WorldModelStore."""

import pytest

from arc_agi_3.executable_model import (
    ExecutableModelCandidate,
    ExecutableModelRegistry,
    source_sha256,
)


def candidate(
    model_id: str, version: int, parent_version: int | None = None
) -> ExecutableModelCandidate:
    return ExecutableModelCandidate(
        model_id=model_id,
        version=version,
        parent_version=parent_version,
        source_artifact_ref="artifacts/x.json",
        source_sha256=source_sha256(f"{model_id}-{version}"),
    )


def test_root_version_needs_no_parent() -> None:
    registry = ExecutableModelRegistry()
    registry.add(candidate("m1", 1))
    assert registry.get("m1", 1).version == 1


def test_a_revision_must_declare_its_direct_parent() -> None:
    registry = ExecutableModelRegistry()
    registry.add(candidate("m1", 1))
    with pytest.raises(ValueError, match="parent_version"):
        registry.add(candidate("m1", 2, parent_version=None))


def test_duplicate_versions_are_rejected() -> None:
    registry = ExecutableModelRegistry()
    registry.add(candidate("m1", 1))
    with pytest.raises(ValueError, match="already exists"):
        registry.add(candidate("m1", 1))


def test_competing_model_ids_coexist() -> None:
    registry = ExecutableModelRegistry()
    registry.add(candidate("m1", 1))
    registry.add(candidate("m2", 1))
    assert registry.model_ids() == ("m1", "m2")
    assert registry.latest("m1").model_id == "m1"
    assert registry.latest("m2").model_id == "m2"


def test_unknown_version_raises_key_error() -> None:
    registry = ExecutableModelRegistry()
    registry.add(candidate("m1", 1))
    with pytest.raises(KeyError):
        registry.get("m1", 2)
