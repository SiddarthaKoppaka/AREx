"""Working memory accepts only typed, evidence-linked model operations."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from arc_agi_3.cognition.scratchpad import ScratchpadStore
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision, ModelResponse
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.scratchpad import ScratchpadUpdates, VerifiedFact
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment


class MemoryModel:
    def __init__(self, evidence: str | None = None) -> None:
        self.evidence = evidence

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "memory-test"}

    def decide(self, context: AgentContext) -> ModelResponse:
        reference = self.evidence or context.recent_event_refs[-1]
        update = ScratchpadUpdates(
            set_objective="Map the movement actions.",
            add_verified_fact=(
                VerifiedFact(
                    fact_id="action-1-local",
                    version=1,
                    fact="Action 1 is available in the initial state.",
                    confidence=0.95,
                    evidence_refs=(reference,),
                ),
            ),
            action_model_updates={"1": "available; movement effect unverified"},
            revise_plan=("Test action 1", "Compare the transition delta"),
        )
        return ModelResponse(
            decision=CognitiveDecision(
                assessment="The initial observation is evidence.",
                intent="Record compact working memory.",
                mode=DecisionMode.STOP,
                scratchpad_updates=update,
            )
        )


class CaptureModel:
    def __init__(self) -> None:
        self.context: AgentContext | None = None

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "capture"}

    def decide(self, context: AgentContext) -> ModelResponse:
        self.context = context
        return ModelResponse(
            decision=CognitiveDecision(
                assessment="Capabilities are already known.",
                intent="Avoid unsupported recovery.",
                mode=DecisionMode.STOP,
            )
        )


class NoRestoreEnvironment(FakeLineEnvironment):
    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "no-restore", "restore_supported": False}


def test_scratchpad_update_is_versioned_and_auditable(tmp_path: Path) -> None:
    config = RunConfig(run_id="memory", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(), MemoryModel())
    result = runner.run()
    scratchpad = runner.io.workspace.scratchpad.current
    assert result.stop_reason == "agent_stop"
    assert scratchpad.version == 2
    assert scratchpad.verified_facts[0].fact_id == "action-1-local"
    assert scratchpad.capabilities["checkpoint_restore_supported"] is True
    updates = [
        event
        for event in runner.events.read()
        if event.event_type is EventType.SCRATCHPAD_UPDATE
    ]
    assert len(updates) == 1


def test_unknown_evidence_ref_rejects_verified_fact(tmp_path: Path) -> None:
    config = RunConfig(run_id="bad-memory", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(), MemoryModel("missing-event"))
    result = runner.run()
    assert result.stop_reason == "failure"
    assert runner.io.workspace.scratchpad.current.verified_facts == ()


def test_scratchpad_budget_rejects_unbounded_updates() -> None:
    store = ScratchpadStore(token_budget=20)
    updates = ScratchpadUpdates(set_objective="x" * 500)
    with pytest.raises(ValueError, match="scratchpad exceeds"):
        store.apply(updates, set())


def test_restore_capability_is_injected_before_first_decision(tmp_path: Path) -> None:
    environment = NoRestoreEnvironment()
    model = CaptureModel()
    config = RunConfig(run_id="capability", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, environment, model)
    runner.run()
    assert model.context is not None
    scratchpad = model.context.working_scratchpad
    assert scratchpad is not None
    assert scratchpad.capabilities["checkpoint_restore_supported"] is False
