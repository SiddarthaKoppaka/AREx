"""Deterministic M1 fixtures."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arc_agi_3.config import EvaluatorConfig, RunConfig
from arc_agi_3.manifest import RunManifest


@pytest.fixture
def run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_id="golden-run",
        experiment_id="foundation-tests",
        output_dir=tmp_path,
        max_turns=4,
        evaluator=EvaluatorConfig(human_action_baselines=(2,)),
    )


@pytest.fixture
def fixed_clock() -> Callable[[], datetime]:
    return lambda: datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def event_ids() -> Callable[[], str]:
    counter = iter(range(1, 1000))
    return lambda: f"event-{next(counter):04d}"


@pytest.fixture
def fixed_manifest(run_config: RunConfig) -> RunManifest:
    return RunManifest(
        run_id=run_config.run_id,
        experiment_id=run_config.experiment_id,
        config_hash=run_config.config_hash,
        config=run_config.model_dump(mode="json", exclude={"output_dir"}),
        harness_version="0.1.0",
        git_commit="test-commit",
        git_dirty=False,
        python_version="3.12.test",
        platform="test-platform",
        model={"adapter": "scripted", "version": "1"},
        environment={"adapter": "fake-line", "version": "1", "target": 2},
    )
