"""Shared fakes for structured-generation robustness tests."""

from types import SimpleNamespace

from pydantic import JsonValue

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.contracts.decision import AgentContext, ModelUsage
from arc_agi_3.testing import FakeLineEnvironment


class ScriptedBackend:
    """Replays one generation per queued item; a queued exception is raised."""

    def __init__(
        self,
        outputs: list[object],
        *,
        input_tokens: int = 2,
        output_tokens: int = 3,
        latency_ms: int = 4,
    ) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []
        self.usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "test", "model": "scripted"}

    def generate(
        self, prompt: str, json_schema: dict[str, object]
    ) -> BackendGeneration:
        self.prompts.append(prompt)
        self.schemas.append(json_schema)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return BackendGeneration(text=str(item), usage=self.usage)


class RecordingReporter:
    def __init__(self) -> None:
        self.stages: list[tuple[str, dict[str, object]]] = []

    def stage(self, name: str, **details: object) -> None:
        self.stages.append((name, details))


def context(turn: int = 1) -> AgentContext:
    return AgentContext(turn=turn, observation=FakeLineEnvironment().reset(), budget={})


class TokenCounter:
    """A backend fake that counts prompt characters as tokens, for compaction tests."""

    def __init__(self, target: int, *, pressure: int = 1) -> None:
        self.config = SimpleNamespace(
            compaction_pressure_start=pressure,
            active_context_target=target,
            max_input_tokens=target + 10_000,
            max_context_tokens=target + 10_128,
            max_new_tokens=64,
            template_and_generation_margin=64,
            model_name="test",
        )

    def input_token_count(self, prompt: str, schema: dict[str, object]) -> int:
        return len(prompt)
