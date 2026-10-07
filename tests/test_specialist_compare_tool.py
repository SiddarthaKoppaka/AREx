"""compare_specialist_reports dispatches through the existing Typed Tool
Interface and fails cleanly without a prior report. See ADR-0009."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_session
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel
from tests.generation_fixtures import ScriptedBackend
from tests.specialist_tool_fixtures import investigate, stop


def test_compare_specialist_reports_fails_cleanly_with_no_prior_report(
    tmp_path: Path,
) -> None:
    specialists = ScriptedBackend([])
    core_agent = ScriptedModel(
        [
            investigate(
                "compare_specialist_reports",
                {"primary_faculty": "critic", "other_faculties": ["hypothesis"]},
            ),
            stop(),
        ]
    )
    config = RunConfig(run_id="tool-compare", experiment_id="test", output_dir=tmp_path)
    session = build_session(
        config, FakeLineEnvironment(), core_agent, specialist_backend=specialists
    )
    session.start(FakeLineEnvironment().reset())
    session.next_action()

    result = next(
        e for e in session.io.events.read() if e.event_type is EventType.TOOL_RESULT
    )
    assert result.payload["result"]["status"] == "failed"
