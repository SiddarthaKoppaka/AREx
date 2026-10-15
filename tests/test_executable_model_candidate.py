"""ExecutableModelCandidate enforces version lineage; hashing is deterministic."""

import pytest

from arc_agi_3.executable_model import ExecutableModelCandidate, source_sha256


def test_source_hash_is_deterministic() -> None:
    assert source_sha256("x = 1") == source_sha256("x = 1")
    assert source_sha256("x = 1") != source_sha256("x = 2")


def candidate(**overrides: object) -> ExecutableModelCandidate:
    fields: dict[str, object] = {
        "model_id": "m1",
        "version": 1,
        "source_artifact_ref": "artifacts/x.json",
        "source_sha256": source_sha256("x = 1"),
    }
    fields.update(overrides)
    return ExecutableModelCandidate(**fields)


def test_a_root_candidate_needs_no_parent_version() -> None:
    assert candidate().parent_version is None


def test_parent_version_must_precede_version() -> None:
    candidate(version=2, parent_version=1)
    with pytest.raises(ValueError, match="parent_version"):
        candidate(version=2, parent_version=2)
    with pytest.raises(ValueError, match="parent_version"):
        candidate(version=2, parent_version=3)
