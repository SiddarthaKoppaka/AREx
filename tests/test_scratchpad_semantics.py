"""next_test is planning memory: revisable, recorded, and never an action lock."""

from pathlib import Path

import pytest

from arc_agi_3.cognition.scratchpad import ScratchpadStore
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.scratchpad import NextTest, ScratchpadUpdates, VerifiedFact
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, ReactiveModel
from tests.epistemic_policies import stop


def test_plan_revision_is_recorded_and_clearable() -> None:
    store = ScratchpadStore()
    store.apply(
        ScratchpadUpdates(set_next_test=NextTest(action_id=1, purpose="a")), set()
    )
    revised = store.apply(
        ScratchpadUpdates(
            set_next_test=NextTest(action_id=2, purpose="b"),
            next_test_revision_reason="Action 1 evidence is already conclusive.",
        ),
        set(),
    )
    revision = revised.plan_revisions[-1]
    assert revision.previous is not None and revision.previous.action_id == 1
    assert revision.revised is not None and revision.revised.action_id == 2
    assert revision.reason == "Action 1 evidence is already conclusive."
    cleared = store.apply(ScratchpadUpdates(clear_next_test=True), set())
    assert cleared.next_test is None and cleared.plan_revisions[-1].revised is None
    unchanged = store.apply(ScratchpadUpdates(set_objective="x"), set())
    assert len(unchanged.plan_revisions) == 2


def plan_then_deviate(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="Plan one probe, but execute another.",
        intent="Show next_test is not executor authority.",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=2),
        scratchpad_updates=ScratchpadUpdates(
            set_next_test=NextTest(action_id=1, purpose="Probe action 1 later.")
        ),
    )


def test_executed_action_may_differ_from_planned_test(tmp_path: Path) -> None:
    model = ReactiveModel([plan_then_deviate, stop])
    config = RunConfig(run_id="plan", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    result = runner.run()
    action = next(e for e in runner.events.read() if e.event_type is EventType.ACTION)
    assert action.payload["action"] == {
        "action_id": 2,
        "data": {},
        "schema_version": "1.0",
    }
    assert action.payload["matches_planned_next_test"] is False
    assert result.metrics.plan_deviations == 1


def test_model_decisions_are_not_evidence_for_verified_facts(tmp_path: Path) -> None:
    def cite_own_decision(context: AgentContext) -> CognitiveDecision:
        decision_id = next(
            e.event_id
            for e in context.recent_events
            if e.event_type is EventType.MODEL_DECISION
        )
        fact = VerifiedFact(
            fact_id="f1",
            version=1,
            fact="My earlier interpretation was right.",
            confidence=0.9,
            evidence_refs=(decision_id,),
        )
        return stop(context).model_copy(
            update={"scratchpad_updates": ScratchpadUpdates(add_verified_fact=(fact,))}
        )

    model = ReactiveModel([plan_then_deviate, cite_own_decision])
    config = RunConfig(run_id="facts", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    with pytest.raises(ValueError, match="observational evidence"):
        runner.run()
