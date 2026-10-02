"""Older contracts and traces stay valid; budgets and prompt reports are exact."""

from pathlib import Path
from types import SimpleNamespace

from pydantic import JsonValue

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import Hypothesis
from arc_agi_3.contracts.decision import CognitiveDecision, ModelUsage
from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.contracts.transition import TransitionEvidence
from arc_agi_3.contracts.verification import StateDelta
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, successful_script


def test_legacy_decision_hypothesis_and_transition_payloads_validate() -> None:
    legacy = {
        "assessment": "a",
        "intent": "i",
        "mode": "execute",
        "action": {"action_id": 1},
        "expected_outcome": {"state": "WIN", "max_changed_cells": 3},
    }
    decision = CognitiveDecision.model_validate(legacy)
    assert decision.expected_outcome is not None
    assert decision.expected_outcome.checkable and decision.experiment is None
    old = Hypothesis.model_validate(
        {"hypothesis_id": "h", "version": 1, "claim": "c", "probability": 0.5}
    )
    assert old.contradicting_refs == () and old.created_turn is None
    delta = {"before_hash": "a", "after_hash": "b", "changed_cells": 2}
    evidence = TransitionEvidence.model_validate({**delta, "metadata_changes": {}})
    assert StateDelta is TransitionEvidence and evidence.cell_changes == ()


class CountingBackend:
    config = SimpleNamespace(
        compaction_pressure_start=10**6,
        active_context_target=10**6,
        max_input_tokens=10**7,
        model_name="counting",
    )

    def __init__(self) -> None:
        self.outputs = [item.model_dump_json() for item in successful_script()]

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "test"}

    def input_token_count(self, prompt: str, schema: dict[str, object]) -> int:
        return len(prompt) // 4

    def generate(self, prompt: str, schema: dict[str, object]) -> BackendGeneration:
        usage = ModelUsage(input_tokens=100, output_tokens=7, latency_ms=1)
        return BackendGeneration(text=self.outputs.pop(0), usage=usage)


def test_budget_metrics_and_prompt_reports_are_recorded(tmp_path: Path) -> None:
    config = RunConfig(run_id="budget", experiment_id="test", output_dir=tmp_path)
    model = StructuredModelAdapter(CountingBackend())
    runner = build_runner(config, FakeLineEnvironment(), model, raise_on_failure=True)
    metrics = runner.run().metrics
    assert (metrics.model_calls, metrics.environment_actions) == (2, 2)
    assert (metrics.input_tokens, metrics.output_tokens) == (200, 14)
    consumed = runner.io.ledger.snapshot().consumed
    assert consumed[Resource.INPUT_TOKENS] == 200 and consumed[Resource.ACTIONS] == 2
    assert (metrics.predictions_matched, metrics.prediction_mismatches) == (2, 0)
    decisions = [
        e for e in runner.events.read() if e.event_type is EventType.MODEL_DECISION
    ]
    report = decisions[-1].payload["prompt"]
    assert isinstance(report, dict) and report["compacted"] is False
    assert report["tokens_before_compaction"] == report["tokens_after_compaction"] > 0
    context = decisions[-1].payload["context"]
    assert isinstance(context, dict) and context["recent_events_included"] > 0
