"""Provider-neutral text generation boundary."""

from typing import Any, Protocol

from pydantic import JsonValue

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.decision import ModelUsage


class BackendGeneration(Contract):
    text: str
    usage: ModelUsage = ModelUsage()


class InferenceBackend(Protocol):
    @property
    def metadata(self) -> dict[str, JsonValue]: ...

    def generate(
        self, prompt: str, json_schema: dict[str, Any]
    ) -> BackendGeneration: ...
