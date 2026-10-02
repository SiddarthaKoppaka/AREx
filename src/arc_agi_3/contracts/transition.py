"""Deterministic, semantics-free evidence about one observation transition."""

from pydantic import Field, JsonValue

from .base import Contract

TRANSITION_EVIDENCE_VERSION = "transition_evidence_v1"


class CellChange(Contract):
    layer: int = Field(ge=0)
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    before: int | None
    after: int | None


class ValueCountChange(Contract):
    value: int
    before: int = Field(ge=0)
    after: int = Field(ge=0)


class ChangedRegion(Contract):
    layer: int = Field(ge=0)
    min_row: int = Field(ge=0)
    min_col: int = Field(ge=0)
    max_row: int = Field(ge=0)
    max_col: int = Field(ge=0)
    changed_cells: int = Field(ge=0)


class ComponentSummary(Contract):
    """A 4-connected equal-value region; no object or role is implied."""

    layer: int = Field(ge=0)
    value: int
    size: int = Field(gt=0)
    min_row: int = Field(ge=0)
    min_col: int = Field(ge=0)
    max_row: int = Field(ge=0)
    max_col: int = Field(ge=0)


class ComponentTranslation(Contract):
    """A disappeared and an appeared component with identical shape and value."""

    layer: int = Field(ge=0)
    value: int
    size: int = Field(gt=0)
    from_row: int = Field(ge=0)
    from_col: int = Field(ge=0)
    d_row: int
    d_col: int


class TransitionEvidence(Contract):
    """Exact counts plus bounded listings; full cells stay retrievable by event ID."""

    evidence_version: str = TRANSITION_EVIDENCE_VERSION
    before_hash: str
    after_hash: str
    changed_cells: int = Field(ge=0)
    metadata_changes: dict[str, tuple[JsonValue, JsonValue]] = Field(
        default_factory=dict
    )
    shape_changed: bool = False
    cell_changes: tuple[CellChange, ...] = ()
    omitted_cell_changes: int = Field(default=0, ge=0)
    value_count_changes: tuple[ValueCountChange, ...] = ()
    changed_regions: tuple[ChangedRegion, ...] = ()
    appeared_components: tuple[ComponentSummary, ...] = ()
    disappeared_components: tuple[ComponentSummary, ...] = ()
    omitted_component_changes: int = Field(default=0, ge=0)
    translations: tuple[ComponentTranslation, ...] = ()
    omitted_translations: int = Field(default=0, ge=0)
