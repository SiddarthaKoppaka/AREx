"""Public decision summaries are concise, persisted, and observable."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from arc_agi_3.adapters.prompts import decision_prompt
from arc_agi_3.config import RunConfig
from arc_agi_3.context.projection import compact_context_event
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.reporting import LiveReporter
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def public_decision() -> CognitiveDecision:
    return CognitiveDecision(
        assessment="One move remains.",
        intent="Finish the level.",
        considered_options=("Advance", "Wait and inspect"),
        decision_summary="Advance because the target is adjacent.",
        observation_summary="The marker is one cell from the target.",
        expected_result="The level reaches WIN.",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
    )


def test_public_decision_fields_are_bounded() -> None:
    values = public_decision().model_dump()
    values["considered_options"] = tuple(str(index) for index in range(6))
    with pytest.raises(ValidationError, match="considered_options"):
        CognitiveDecision.model_validate(values)
    values = public_decision().model_dump()
    values["decision_summary"] = "x" * 513
    with pytest.raises(ValidationError, match="decision_summary"):
        CognitiveDecision.model_validate(values)


def test_prompt_requests_public_summary_without_private_reasoning() -> None:
    context = AgentContext(
        turn=1, observation=FakeLineEnvironment(target=1).reset(), budget={}
    )
    prompt = decision_prompt(context)
    assert "concise public decision rationale and summaries" in prompt
    assert "Do not provide private chain-of-thought or hidden reasoning" in prompt


def test_public_summary_is_traced_reported_and_compacted(tmp_path: Path) -> None:
    reporter = LiveReporter("readable", tmp_path / "console.log")
    runner = build_runner(
        RunConfig(run_id="public", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(target=1),
        ScriptedModel([public_decision()]),
        reporter=reporter,
    )
    assert runner.run().stop_reason == "terminal"
    event = next(
        item
        for item in runner.events.read()
        if item.event_type is EventType.MODEL_DECISION
    )
    decision = event.payload["decision"]
    assert decision["considered_options"] == ["Advance", "Wait and inspect"]
    assert decision["decision_summary"] == "Advance because the target is adjacent."
    compact = compact_context_event(event).payload
    assert compact["observation_summary"] == ("The marker is one cell from the target.")
    assert "decision" not in compact and "action" not in compact
    log = (tmp_path / "console.log").read_text()
    assert '"expected_result":"The level reaches WIN."' in log


def test_absent_public_summaries_do_not_expand_trace(tmp_path: Path) -> None:
    decision = CognitiveDecision(
        assessment="advance",
        intent="finish",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
    )
    runner = build_runner(
        RunConfig(run_id="minimal", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(target=1),
        ScriptedModel([decision]),
    )
    runner.run()
    event = next(
        item
        for item in runner.events.read()
        if item.event_type is EventType.MODEL_DECISION
    )
    assert not set(event.payload["decision"]) & {
        "considered_options",
        "decision_summary",
        "observation_summary",
        "expected_result",
    }
