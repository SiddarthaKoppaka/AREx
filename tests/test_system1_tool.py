"""consult_system1 dispatches through the Typed Tool Interface in suggest
mode; the Core Agent's own next decision is independent either way. See
ADR-0010. Authorization/safety boundary tests live in
test_system1_authorization.py."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_session
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel
from tests.specialist_tool_fixtures import investigate, stop
from tests.system1_fixtures import (
    FakeSystem1Backend,
    result_selecting,
    two_candidate_request,
)


def test_suggest_mode_records_a_decision_and_never_acts(tmp_path: Path) -> None:
    system1 = FakeSystem1Backend([result_selecting("move-down")])
    core_agent = ScriptedModel(
        [investigate("consult_system1", two_candidate_request()), stop()]
    )
    config = RunConfig(
        run_id="system1-suggest", experiment_id="test", output_dir=tmp_path
    )
    session = build_session(
        config, FakeLineEnvironment(), core_agent, system1_backend=system1
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    events = session.io.events.read()
    kinds = [e.event_type for e in events]
    assert kinds.count(EventType.SYSTEM1_DECISION) == 1
    assert kinds.count(EventType.ACTION) == 0
    result = next(e for e in events if e.event_type is EventType.TOOL_RESULT)
    assert result.payload["result"]["status"] == "complete"
    assert result.payload["result"]["output"]["system1"]["selected_candidate_id"] == (
        "move-down"
    )


def test_core_agent_override_is_preserved_in_trace(tmp_path: Path) -> None:
    # System-1 recommends inspect-region; the Core Agent's own trace record
    # of what it does next is independent and untouched either way - this
    # proves the recommendation never silently becomes the final decision.
    system1 = FakeSystem1Backend([result_selecting("inspect-region")])
    core_agent = ScriptedModel(
        [investigate("consult_system1", two_candidate_request()), stop()]
    )
    config = RunConfig(
        run_id="system1-override", experiment_id="test", output_dir=tmp_path
    )
    session = build_session(
        config, FakeLineEnvironment(), core_agent, system1_backend=system1
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    events = session.io.events.read()
    decision = next(e for e in events if e.event_type is EventType.SYSTEM1_DECISION)
    assert decision.payload["result"]["selected_candidate_id"] == "inspect-region"
    decisions = [e for e in events if e.event_type is EventType.MODEL_DECISION]
    assert decisions[-1].payload["decision"]["mode"] == "stop"


def test_no_system1_backend_configured_fails_cleanly(tmp_path: Path) -> None:
    core_agent = ScriptedModel(
        [investigate("consult_system1", two_candidate_request()), stop()]
    )
    config = RunConfig(run_id="system1-none", experiment_id="test", output_dir=tmp_path)
    session = build_session(config, FakeLineEnvironment(), core_agent)
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    result = next(
        e for e in session.io.events.read() if e.event_type is EventType.TOOL_RESULT
    )
    assert result.payload["result"]["status"] == "failed"
