"""Exact retrieval and provenance-bearing context records."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .base import Contract
from .enums import EventType


class RetrievalQuery(Contract):
    event_types: tuple[EventType, ...] = ()
    event_ids: tuple[str, ...] = ()
    causal_ref: str | None = None
    component: str | None = None
    literal_text: str | None = None
    min_sequence: int | None = Field(default=None, ge=0)
    max_sequence: int | None = Field(default=None, ge=0)
    order: Literal["ascending", "descending"] = "ascending"
    limit: int = Field(default=20, gt=0, le=200)

    @model_validator(mode="after")
    def valid_range(self) -> "RetrievalQuery":
        if (
            self.min_sequence is not None
            and self.max_sequence is not None
            and self.min_sequence > self.max_sequence
        ):
            raise ValueError("min_sequence must not exceed max_sequence")
        return self


class ContextEvent(Contract):
    event_id: str
    event_hash: str
    sequence: int = Field(ge=0)
    event_type: EventType
    component: str
    causal_refs: tuple[str, ...]
    payload: dict[str, JsonValue]


class RetrievalResult(Contract):
    query: RetrievalQuery
    matches: tuple[ContextEvent, ...]
