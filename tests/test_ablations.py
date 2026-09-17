"""Ablations remove substrate capabilities without replacing LM intent."""

from arc_agi_3.ablations import PRESETS, AblationPreset, apply_preset
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import Hypothesis
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def test_presets_are_explicit_and_hash_visible(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = RunConfig(run_id="r", experiment_id="e", output_dir=tmp_path)
    thin = apply_preset(base, AblationPreset.THIN)
    assert thin.ablations == PRESETS[AblationPreset.THIN]
    assert not thin.ablations.search
    assert thin.config_hash != base.config_hash


def test_disabled_capability_fails_visibly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = RunConfig(run_id="r", experiment_id="e", output_dir=tmp_path, max_turns=1)
    config = apply_preset(base, AblationPreset.THIN)
    decision = CognitiveDecision(
        assessment="candidate",
        intent="record a hypothesis",
        mode=DecisionMode.INVESTIGATE,
        hypothesis_proposals=(
            Hypothesis(hypothesis_id="h", version=1, claim="test", probability=0.5),
        ),
    )
    runner = build_runner(config, FakeLineEnvironment(), ScriptedModel([decision]))
    assert runner.run().stop_reason == "failure"
    failures = [
        event for event in runner.events.read() if event.event_type is EventType.FAILURE
    ]
    assert "hypotheses" in str(failures[0].payload.get("message"))
