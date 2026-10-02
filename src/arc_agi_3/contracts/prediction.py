"""LM-authored observable predictions and experiment framing.

Every field is optional so ordinary exploitative actions stay lightweight. Only
the observable claims are machine-checked; prose fields are recorded verbatim.
"""

from pydantic import Field

from .base import Contract
from .enums import EnvironmentState

CHECKABLE_FIELDS = (
    "observation_hash",
    "state",
    "min_levels_completed",
    "min_changed_cells",
    "max_changed_cells",
    "no_op",
    "cell_values",
    "translations",
)


class CellExpectation(Contract):
    layer: int = Field(default=0, ge=0)
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: int


class TranslationExpectation(Contract):
    """Some component of `value` is expected to shift by (d_row, d_col)."""

    layer: int = Field(default=0, ge=0)
    value: int
    d_row: int
    d_col: int
    from_row: int | None = Field(default=None, ge=0)
    from_col: int | None = Field(default=None, ge=0)


class ExpectedOutcome(Contract):
    prediction_id: str | None = None
    hypothesis_ids: tuple[str, ...] = ()
    observation_hash: str | None = None
    state: EnvironmentState | None = None
    min_levels_completed: int | None = Field(default=None, ge=0)
    min_changed_cells: int | None = Field(default=None, ge=0)
    max_changed_cells: int | None = Field(default=None, ge=0)
    no_op: bool | None = None
    cell_values: tuple[CellExpectation, ...] = Field(default=(), max_length=16)
    translations: tuple[TranslationExpectation, ...] = Field(default=(), max_length=8)
    description: str | None = Field(default=None, max_length=240)
    supports_if: str | None = Field(default=None, max_length=240)
    contradicts_if: str | None = Field(default=None, max_length=240)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def checkable(self) -> bool:
        return any(getattr(self, name) not in (None, ()) for name in CHECKABLE_FIELDS)


class Experiment(Contract):
    """Marks an action as an investigation; the harness never selects one."""

    experiment_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=240)
    hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=8)
    repeat_justification: str | None = Field(default=None, max_length=240)
