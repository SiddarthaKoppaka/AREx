"""Exact state comparison and LM-requested outcome checks."""

from typing import Any

from arc_agi_3.contracts.decision import ExpectedOutcome
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.verification import StateDelta, VerificationResult


def _cells(observation: Observation) -> dict[tuple[int, int, int], int]:
    return {
        (layer_index, row_index, column_index): value
        for layer_index, layer in enumerate(observation.frame)
        for row_index, row in enumerate(layer)
        for column_index, value in enumerate(row)
    }


def compare_observations(before: Observation, after: Observation) -> StateDelta:
    left, right = _cells(before), _cells(after)
    changed = sum(left.get(key) != right.get(key) for key in left.keys() | right.keys())
    metadata: dict[str, tuple[Any, Any]] = {}
    for name in (
        "state",
        "levels_completed",
        "win_levels",
        "available_actions",
        "full_reset",
    ):
        old, new = getattr(before, name), getattr(after, name)
        if old != new:
            old = list(old) if isinstance(old, tuple) else old
            new = list(new) if isinstance(new, tuple) else new
            metadata[name] = (old, new)
    return StateDelta(
        before_hash=before.observation_hash,
        after_hash=after.observation_hash,
        changed_cells=changed,
        metadata_changes=metadata,
    )


def verify_outcome(
    expected: ExpectedOutcome | None,
    before: Observation,
    after: Observation,
) -> VerificationResult:
    delta = compare_observations(before, after)
    if expected is None:
        return VerificationResult(checked=False, passed=True, delta=delta)
    mismatches: list[str] = []
    if (
        expected.observation_hash
        and expected.observation_hash != after.observation_hash
    ):
        mismatches.append("observation_hash")
    if expected.state and expected.state != after.state:
        mismatches.append("state")
    if (
        expected.min_levels_completed is not None
        and after.levels_completed < expected.min_levels_completed
    ):
        mismatches.append("levels_completed")
    if (
        expected.max_changed_cells is not None
        and delta.changed_cells > expected.max_changed_cells
    ):
        mismatches.append("changed_cells")
    return VerificationResult(
        checked=True, passed=not mismatches, mismatches=tuple(mismatches), delta=delta
    )
