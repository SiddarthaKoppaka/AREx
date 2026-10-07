"""consult_specialist dispatches through the existing Typed Tool Interface;
a core agent that requests nothing runs zero specialists. See ADR-0009."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.runtime import build_session
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script
from tests.generation_fixtures import ScriptedBackend
from tests.specialist_tool_fixtures import HYPOTHESIS_REPORT, investigate, stop


def test_consult_specialist_runs_exactly_one_report_and_charges_budget(
    tmp_path: Path,
) -> None:
    specialists = ScriptedBackend([HYPOTHESIS_REPORT], input_tokens=10, output_tokens=5)
    core_agent = ScriptedModel(
        [investigate("consult_specialist", {"faculty": "hypothesis"}), stop()]
    )
    config = RunConfig(run_id="tool-consult", experiment_id="test", output_dir=tmp_path)
    session = build_session(
        config, FakeLineEnvironment(), core_agent, specialist_backend=specialists
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    events = session.io.events.read()
    kinds = [e.event_type for e in events]
    assert kinds.count(EventType.SPECIALIST_REPORT) == 1
    assert kinds.count(EventType.TOOL_RESULT) == 1
    result = next(e for e in events if e.event_type is EventType.TOOL_RESULT)
    assert result.payload["result"]["status"] == "complete"
    assert session.io.ledger.consumed[Resource.INPUT_TOKENS] >= 10


def test_core_agent_requesting_nothing_runs_zero_specialists(tmp_path: Path) -> None:
    specialists = ScriptedBackend([])
    core_agent = ScriptedModel(successful_script())
    config = RunConfig(run_id="tool-idle", experiment_id="test", output_dir=tmp_path)
    session = build_session(
        config, FakeLineEnvironment(), core_agent, specialist_backend=specialists
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    kinds = [e.event_type for e in session.io.events.read()]
    assert kinds.count(EventType.SPECIALIST_REPORT) == 0
    assert specialists.prompts == []


def test_consult_specialist_without_a_configured_exocortex_fails_cleanly(
    tmp_path: Path,
) -> None:
    core_agent = ScriptedModel(
        [investigate("consult_specialist", {"faculty": "hypothesis"}), stop()]
    )
    config = RunConfig(run_id="tool-none", experiment_id="test", output_dir=tmp_path)
    session = build_session(config, FakeLineEnvironment(), core_agent)
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    result = next(
        e for e in session.io.events.read() if e.event_type is EventType.TOOL_RESULT
    )
    assert result.payload["result"]["status"] == "failed"
