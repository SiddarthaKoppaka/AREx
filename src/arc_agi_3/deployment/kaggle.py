"""Thin notebook-facing entrypoint; adapters remain independently replaceable."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.evaluation import EvaluationMetrics
from arc_agi_3.runtime.factory import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def offline_smoke(
    output: Path, run_id: str = "kaggle-offline-smoke"
) -> EvaluationMetrics:
    config = RunConfig(
        run_id=run_id,
        experiment_id="kaggle-offline-smoke",
        output_dir=output,
    )
    return (
        build_runner(config, FakeLineEnvironment(), ScriptedModel(successful_script()))
        .run()
        .metrics
    )
