"""Evaluation exposes real generation cost, and budgets charge every attempt."""

import json
from pathlib import Path

from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import Resource
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, successful_script
from tests.generation_fixtures import ScriptedBackend

VALID = [json.dumps(item.model_dump(mode="json")) for item in successful_script()]


def test_generation_metrics_count_primary_and_repair_attempts(tmp_path: Path) -> None:
    backend = ScriptedBackend(["not json", VALID[0], VALID[1]])
    config = RunConfig(run_id="metrics", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), StructuredModelAdapter(backend)
    )
    result = runner.run()
    metrics = result.metrics
    assert (metrics.primary_generations, metrics.repair_generations) == (2, 1)
    assert metrics.backend_generations == 3
    assert metrics.failed_generations == 1
    assert metrics.structured_output_failures == 0
    assert metrics.avg_backend_generations_per_decision == 1.5
    assert metrics.model_calls == 2


def test_budget_charges_every_backend_generation_including_repairs(
    tmp_path: Path,
) -> None:
    backend = ScriptedBackend(
        ["not json", VALID[0], VALID[1]], input_tokens=10, output_tokens=5
    )
    config = RunConfig(run_id="budget-gen", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), StructuredModelAdapter(backend)
    )
    result = runner.run()
    consumed = runner.io.ledger.consumed
    assert consumed[Resource.INPUT_TOKENS] == 30
    assert consumed[Resource.OUTPUT_TOKENS] == 15
    assert result.metrics.input_tokens == 30
    assert result.metrics.repair_prompt_tokens == 10


def test_structured_output_failure_is_counted_distinctly(tmp_path: Path) -> None:
    backend = ScriptedBackend(["x", "y", "z"])
    config = RunConfig(
        run_id="failure-metrics", experiment_id="test", output_dir=tmp_path
    )
    runner = build_runner(
        config, FakeLineEnvironment(), StructuredModelAdapter(backend, max_repairs=2)
    )
    result = runner.run()
    assert result.stop_reason == "failure"
    assert result.metrics.structured_output_failures == 1
    assert result.metrics.backend_generations == 3
    assert result.metrics.failed_generations == 3
