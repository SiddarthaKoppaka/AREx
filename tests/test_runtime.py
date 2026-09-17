"""End-to-end foundation slice and agency boundary regression."""

from collections.abc import Callable
from datetime import datetime

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.manifest import RunManifest
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def test_fake_run_is_complete_and_auditable(
    run_config: RunConfig,
    fixed_clock: Callable[[], datetime],
    event_ids: Callable[[], str],
    fixed_manifest: RunManifest,
) -> None:
    runner = build_runner(
        run_config,
        FakeLineEnvironment(),
        ScriptedModel(successful_script()),
        clock=fixed_clock,
        id_factory=event_ids,
        manifest=fixed_manifest,
    )
    result = runner.run()
    events = runner.events.read()
    assert result.stop_reason == "terminal"
    assert result.metrics.succeeded
    assert result.metrics.environment_actions == 2
    assert result.metrics.rhae_environment_score == 1.0
    assert audit_agency_boundary(events) == ()


def test_invalid_lm_action_is_stopped_not_replaced(
    run_config: RunConfig,
    fixed_manifest: RunManifest,
) -> None:
    invalid = CognitiveDecision(
        assessment="fixture asks for an unavailable action",
        intent="test rejection",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=7),
    )
    config = run_config.model_copy(update={"run_id": "invalid-run"})
    manifest = fixed_manifest.model_copy(update={"run_id": "invalid-run"})
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel([invalid]), manifest=manifest
    )
    result = runner.run()
    event_types = [event.event_type for event in runner.events.read()]
    assert result.stop_reason == "failure"
    assert EventType.ACTION not in event_types
    assert EventType.FAILURE in event_types
