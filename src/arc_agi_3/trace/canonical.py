"""Canonical serialization used by hashes and golden traces."""

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        default=_default,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()
