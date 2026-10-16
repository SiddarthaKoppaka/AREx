"""PR-12 (idea C): a specialist exhausting its repair budget fails the
tool, not the episode. Mirrors `tests/test_specialist_tools.py`'s exact
build_session/ScriptedBackend pattern."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType, Resource
from arc_agi_3.runtime import build_runner, build_session
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel
from tests.generation_fixtures import ScriptedBackend
from tests.specialist_tool_fixtures import investigate, stop


def test_specialist_exhaustion_fails_the_tool_not_the_episode(tmp_path: Path) -> None:
    specialists = ScriptedBackend(["x", "y", "z"])
    core_agent = ScriptedModel(
        [investigate("consult_specialist", {"faculty": "hypothesis"}), stop()]
    )
    config = RunConfig(run_id="tool-exhaust", experiment_id="test", output_dir=tmp_path)
    session = build_session(
        config, FakeLineEnvironment(), core_agent, specialist_backend=specialists
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    events = session.io.events.read()
    kinds = [e.event_type for e in events]
    assert kinds.count(EventType.FAILURE) == 0
    result = next(e for e in events if e.event_type is EventType.TOOL_RESULT)
    assert result.payload["result"]["status"] == "failed"
    assert result.payload["result"]["output"]["error_type"] == "StructuredOutputError"
    tool_request = next(e for e in events if e.event_type is EventType.TOOL_REQUEST)
    budget_events = [e for e in events if e.event_type is EventType.BUDGET]
    assert any(tool_request.event_id in e.causal_refs for e in budget_events)
    assert session.io.ledger.consumed[Resource.INPUT_TOKENS] >= 2


def test_specialist_exhaustion_counts_as_a_specialist_structured_output_failure(
    tmp_path: Path,
) -> None:
    specialists = ScriptedBackend(["x", "y", "z"])
    core_agent = ScriptedModel(
        [investigate("consult_specialist", {"faculty": "hypothesis"}), stop()]
    )
    config = RunConfig(
        run_id="tool-exhaust-2", experiment_id="test", output_dir=tmp_path
    )
    runner = build_runner(
        config, FakeLineEnvironment(), core_agent, specialist_backend=specialists
    )
    result = runner.run()
    assert result.metrics.specialist_structured_output_failures == 1
    assert result.metrics.failures == 0
