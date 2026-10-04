"""Load a RunConfig from a TOML or JSON profile file."""

import json
import tomllib
from pathlib import Path
from typing import Any

from .config import RunConfig


def load_config(path: Path) -> RunConfig:
    if path.suffix == ".toml":
        with path.open("rb") as stream:
            values: dict[str, Any] = tomllib.load(stream)
    elif path.suffix == ".json":
        values = json.loads(path.read_text())
    else:
        raise ValueError("configuration must be TOML or JSON")
    return RunConfig.model_validate(values)
