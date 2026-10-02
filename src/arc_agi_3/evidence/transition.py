"""Build bounded TransitionEvidence from two observations."""

from typing import Any

from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.transition import TransitionEvidence

from .cells import cell_changes, changed_regions, shape, value_count_changes
from .components import component_changes

METADATA_FIELDS = (
    "state",
    "levels_completed",
    "win_levels",
    "available_actions",
    "full_reset",
)


def metadata_changes(before: Observation, after: Observation) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for name in METADATA_FIELDS:
        old, new = getattr(before, name), getattr(after, name)
        if old != new:
            old = list(old) if isinstance(old, tuple) else old
            new = list(new) if isinstance(new, tuple) else new
            changes[name] = (old, new)
    return changes


def build_transition(
    before: Observation,
    after: Observation,
    *,
    cell_limit: int = 24,
    component_limit: int = 6,
) -> TransitionEvidence:
    """Exact counts with bounded listings; omitted counts are always explicit."""
    changes = cell_changes(before.frame, after.frame)
    evidence = TransitionEvidence(
        before_hash=before.observation_hash,
        after_hash=after.observation_hash,
        changed_cells=len(changes),
        metadata_changes=metadata_changes(before, after),
        shape_changed=shape(before.frame) != shape(after.frame),
        cell_changes=tuple(changes[:cell_limit]),
        omitted_cell_changes=max(0, len(changes) - cell_limit),
    )
    if not changes:
        return evidence
    gone, came, moved = component_changes(before.frame, after.frame)
    omitted = max(0, len(gone) - component_limit) + max(0, len(came) - component_limit)
    return evidence.model_copy(
        update={
            "value_count_changes": value_count_changes(before.frame, after.frame),
            "changed_regions": changed_regions(changes),
            "disappeared_components": tuple(gone[:component_limit]),
            "appeared_components": tuple(came[:component_limit]),
            "omitted_component_changes": omitted,
            "translations": tuple(moved[:component_limit]),
            "omitted_translations": max(0, len(moved) - component_limit),
        }
    )
