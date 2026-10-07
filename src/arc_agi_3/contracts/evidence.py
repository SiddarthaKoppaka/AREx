"""LM-selected, bounded views over canonical event evidence."""

from typing import Literal

from pydantic import Field, model_validator

from .base import Contract

EvidenceView = Literal[
    "metadata", "summary", "transition_delta", "rle_frame", "region", "full"
]


class FrameRegion(Contract):
    layer: int = Field(ge=0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0, le=64)
    height: int = Field(gt=0, le=64)


class EvidenceQuery(Contract):
    event_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    view: EvidenceView
    purpose: str = Field(min_length=1, max_length=240)
    token_budget: int = Field(
        ge=128,
        le=65536,
        description=(
            "A serialized UTF-8 byte upper bound, not an LM tokenizer token "
            "count — see the response's budget_method field."
        ),
    )
    region: FrameRegion | None = None

    @model_validator(mode="after")
    def region_matches_view(self) -> "EvidenceQuery":
        if (self.view == "region") != (self.region is not None):
            raise ValueError("region coordinates are required only for region view")
        return self


class FrameRegionRequest(Contract):
    """The dedicated, non-ambiguous contract for the inspect_frame_region tool.

    Deliberately not EvidenceQuery: that generic contract's nested `region`
    (layer/x/y/width/height) was being reused for this tool via a forced
    `view="region"`, which the model had no way to discover and tried to
    reverse-engineer from validation errors across several turns.
    """

    event_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1, max_length=240)
    token_budget: int = Field(ge=128, le=65536)
    layer: int = Field(default=0, ge=0)
    min_row: int = Field(ge=0)
    max_row: int = Field(ge=0)
    min_col: int = Field(ge=0)
    max_col: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered_bounds(self) -> "FrameRegionRequest":
        if self.max_row < self.min_row or self.max_col < self.min_col:
            raise ValueError("max_row/max_col must not be less than min_row/min_col")
        return self
