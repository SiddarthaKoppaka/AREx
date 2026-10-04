"""A malformed first response is repaired cheaply and the episode continues."""

import json
from pathlib import Path

from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, successful_script
from tests.generation_fixtures import ScriptedBackend

VALID = [json.dumps(item.model_dump(mode="json")) for item in successful_script()]


def test_fake_episode_recovers_from_a_malformed_first_decision(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        ["```json\n{not valid\n```", VALID[0], VALID[1]], latency_ms=5
    )
    config = RunConfig(run_id="e2e-repair", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config,
        FakeLineEnvironment(),
        StructuredModelAdapter(backend),
        raise_on_failure=True,
    )
    result = runner.run()

    assert result.stop_reason == "terminal"
    assert result.metrics.succeeded is True
    assert result.metrics.environment_actions == 2
    assert result.metrics.model_calls == 2
    assert (result.metrics.primary_generations, result.metrics.repair_generations) == (
        2,
        1,
    )
    assert result.metrics.structured_output_failures == 0

    events = runner.events.read()
    decisions = [e for e in events if e.event_type is EventType.MODEL_DECISION]
    assert len(decisions[0].payload["attempts"]) == 2
    assert decisions[0].payload["attempts"][0]["role"] == "primary"
    assert decisions[0].payload["attempts"][1]["role"] == "repair"
    assert len(decisions[1].payload["attempts"]) == 1
    assert audit_agency_boundary(events) == ()
    assert backend.prompts[1] != backend.prompts[0]
    assert "CURRENT_OBSERVATION" not in backend.prompts[1]
