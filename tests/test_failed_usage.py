"""Failed structured generations remain visible in resource accounting."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.config import BudgetConfig, RunConfig
from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment


class InvalidBackend:
    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "fake"}

    def generate(
        self, prompt: str, json_schema: dict[str, object]
    ) -> BackendGeneration:
        return BackendGeneration(
            text="invalid output",
            usage=ModelUsage(input_tokens=7, output_tokens=3, latency_ms=11),
        )


def test_failed_structured_generations_retain_usage(tmp_path: Path) -> None:
    config = RunConfig(
        run_id="failed-usage",
        experiment_id="budgets",
        output_dir=tmp_path,
        budget=BudgetConfig(limits={Resource.MODEL_CALLS: 4, Resource.ACTIONS: 4}),
    )
    runner = build_runner(
        config, FakeLineEnvironment(), StructuredModelAdapter(InvalidBackend())
    )
    result = runner.run()
    assert result.stop_reason == "failure"
    assert (
        result.metrics.model_calls,
        result.metrics.input_tokens,
        result.metrics.output_tokens,
    ) == (1, 14, 6)
    assert runner.io.ledger.consumed[Resource.WALL_TIME_MS] == 22
    assert result.metrics.wall_time_ms == 22
    failure = next(
        event for event in runner.events.read() if event.event_type is EventType.FAILURE
    )
    assert failure.payload["details"]["generation_attempts"] == 2
    assert "invalid output" not in str(failure.payload)


def test_opt_in_preview_is_bounded_artifact_outside_trace(tmp_path: Path) -> None:
    config = RunConfig(run_id="debug-output", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config,
        FakeLineEnvironment(),
        StructuredModelAdapter(InvalidBackend(), persist_invalid_output=True),
    )
    runner.run()
    failure = next(
        event for event in runner.events.read() if event.event_type is EventType.FAILURE
    )
    assert "invalid output" not in str(failure.payload)
    artifact = tmp_path / "debug-output" / "invalid_model_outputs.json"
    assert "invalid output" in artifact.read_text()
