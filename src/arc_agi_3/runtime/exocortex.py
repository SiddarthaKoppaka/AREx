"""The harness-side capability bundle a core agent may consult. See ADR-0009.

Not a Contract: it holds a live backend, never serialized or hashed, the
same way `EpisodeIO` itself is a runtime object rather than a record.
"""

from dataclasses import dataclass

from pydantic import JsonValue

from arc_agi_3.adapters.inference import InferenceBackend
from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.exocortex_config import ExoCortexConfig


@dataclass(frozen=True)
class ExoCortex:
    backend: InferenceBackend
    config: ExoCortexConfig


def model_metadata(
    model: ModelAdapter, exocortex: ExoCortex | None
) -> dict[str, JsonValue]:
    if exocortex is None:
        return model.metadata
    return {
        **model.metadata,
        "exocortex": {
            "backend": exocortex.backend.metadata,
            "faculties": list(exocortex.config.faculties),
        },
    }
