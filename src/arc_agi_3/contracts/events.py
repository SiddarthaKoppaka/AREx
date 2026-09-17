"""Immutable event and checkpoint envelopes."""

from datetime import datetime

from pydantic import Field, JsonValue

from .base import Contract
from .enums import EventType


class EventEnvelope(Contract):
    event_id: str
    run_id: str
    episode_id: str
    branch_id: str
    step_id: int = Field(ge=0)
    sequence: int = Field(ge=0)
    timestamp: datetime
    event_type: EventType
    component: str
    causal_refs: tuple[str, ...] = ()
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    budget: dict[str, JsonValue] = Field(default_factory=dict)
    previous_event_hash: str | None = None
    event_hash: str


class CheckpointRecord(Contract):
    checkpoint_id: str
    run_id: str
    episode_id: str
    branch_id: str
    step_id: int = Field(ge=0)
    last_event_hash: str
    observation: dict[str, JsonValue]
    budget: dict[str, JsonValue]
    environment_state: dict[str, JsonValue] | None = None
    cognitive_state: dict[str, JsonValue] = Field(default_factory=dict)
    checksum: str
