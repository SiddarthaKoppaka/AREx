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
    token_budget: int = Field(ge=128, le=65536)
    region: FrameRegion | None = None

    @model_validator(mode="after")
    def region_matches_view(self) -> "EvidenceQuery":
        if (self.view == "region") != (self.region is not None):
            raise ValueError("region coordinates are required only for region view")
        return self
