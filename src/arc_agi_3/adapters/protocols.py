"""Framework-independent adapter protocols."""

from typing import Protocol

from pydantic import JsonValue

from arc_agi_3.contracts.decision import AgentContext, ModelResponse
from arc_agi_3.contracts.observation import Action, Observation


class EnvironmentAdapter(Protocol):
    @property
    def metadata(self) -> dict[str, JsonValue]: ...

    def reset(self) -> Observation: ...

    def step(self, action: Action) -> Observation: ...

    def checkpoint(self) -> dict[str, JsonValue] | None: ...

    def restore(self, state: dict[str, JsonValue]) -> Observation: ...


class ModelAdapter(Protocol):
    @property
    def metadata(self) -> dict[str, JsonValue]: ...

    def decide(self, context: AgentContext) -> ModelResponse: ...
