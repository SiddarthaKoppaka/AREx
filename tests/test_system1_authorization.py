"""System-1 can never substitute, invent, or directly execute a candidate;
executed actions still trace to exactly one EXECUTE decision. ADR-0010."""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner, build_session
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel
from tests.specialist_tool_fixtures import investigate, stop
from tests.system1_fixtures import (
    FakeSystem1Backend,
    result_selecting,
    two_candidate_request,
)


def test_unauthorized_selected_candidate_fails_cleanly_not_a_crash(
    tmp_path: Path,
) -> None:
    system1 = FakeSystem1Backend([result_selecting("not-a-real-candidate")])
    core_agent = ScriptedModel(
        [investigate("consult_system1", two_candidate_request()), stop()]
    )
    config = RunConfig(
        run_id="system1-unauth", experiment_id="test", output_dir=tmp_path
    )
    session = build_session(
        config, FakeLineEnvironment(), core_agent, system1_backend=system1
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    result = next(
        e for e in session.io.events.read() if e.event_type is EventType.TOOL_RESULT
    )
    assert result.payload["result"]["status"] == "failed"
    assert "unauthorized" in result.payload["result"]["output"]["message"]


def test_execute_one_is_rejected_cleanly_for_now(tmp_path: Path) -> None:
    request = two_candidate_request()
    request["execution_mode"] = "execute_one"
    system1 = FakeSystem1Backend([result_selecting("move-down")])
    core_agent = ScriptedModel([investigate("consult_system1", request), stop()])
    config = RunConfig(
        run_id="system1-exec-one", experiment_id="test", output_dir=tmp_path
    )
    session = build_session(
        config, FakeLineEnvironment(), core_agent, system1_backend=system1
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    result = next(
        e for e in session.io.events.read() if e.event_type is EventType.TOOL_RESULT
    )
    assert result.payload["result"]["status"] == "failed"
    assert system1.requests == []


def test_executed_action_still_traces_to_one_execute_decision(tmp_path: Path) -> None:
    system1 = FakeSystem1Backend([result_selecting("move-down")])
    core_agent = ScriptedModel(
        [
            investigate("consult_system1", two_candidate_request()),
            CognitiveDecision(
                assessment="advance",
                intent="finish",
                mode=DecisionMode.EXECUTE,
                action=Action(action_id=1),
            ),
        ]
    )
    config = RunConfig(
        run_id="system1-audit", experiment_id="test", output_dir=tmp_path
    )
    runner = build_runner(
        config,
        FakeLineEnvironment(target=1),
        core_agent,
        system1_backend=system1,
        raise_on_failure=True,
    )
    runner.run()
    assert audit_agency_boundary(runner.events.read()) == ()
