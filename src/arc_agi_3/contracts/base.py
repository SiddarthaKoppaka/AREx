"""Shared validation policy for persisted contracts."""

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.0"


class Contract(BaseModel):
    """Strict immutable record suitable for hashing and replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = SCHEMA_VERSION
