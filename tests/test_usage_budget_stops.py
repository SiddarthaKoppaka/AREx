"""Measured generation usage can cleanly exhaust token and time budgets."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.config import BudgetConfig, RunConfig
from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment


class MeteredBackend:
    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "fake"}

    def generate(
        self, prompt: str, json_schema: dict[str, object]
    ) -> BackendGeneration:
        return BackendGeneration(
            text='{"assessment":"act","intent":"advance","mode":"execute",'
            '"action":{"action_id":1}}',
            usage=ModelUsage(input_tokens=7, output_tokens=3, latency_ms=11),
        )


@pytest.mark.parametrize(
    ("resource", "reason"),
    [
        (Resource.INPUT_TOKENS, "input_token_budget_exhausted"),
        (Resource.OUTPUT_TOKENS, "output_token_budget_exhausted"),
        (Resource.WALL_TIME_MS, "wall_time_budget_exhausted"),
    ],
)
def test_generation_overrun_stops_before_action(
    tmp_path: Path, resource: Resource, reason: str
) -> None:
    config = RunConfig(
        run_id=resource.value,
        experiment_id="usage-budget",
        output_dir=tmp_path,
        budget=BudgetConfig(limits={resource: 1, Resource.MODEL_CALLS: 2}),
    )
    runner = build_runner(
        config, FakeLineEnvironment(), StructuredModelAdapter(MeteredBackend())
    )
    result = runner.run()
    kinds = [event.event_type for event in runner.events.read()]
    assert result.stop_reason == reason
    assert EventType.RUN_FINISHED in kinds
    assert EventType.FAILURE not in kinds
    assert EventType.ACTION not in kinds
    assert (
        result.metrics.model_calls,
        result.metrics.input_tokens,
        result.metrics.output_tokens,
    ) == (1, 7, 3)
