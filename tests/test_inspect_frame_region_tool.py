"""inspect_frame_region executes against its own dedicated contract."""

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime.factory import build_session
from arc_agi_3.runtime.tool_handlers import execute_tool
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def test_inspect_frame_region_returns_the_exact_requested_crop(tmp_path) -> None:
    session = build_session(
        RunConfig(run_id="region-tool", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(),
        ScriptedModel([]),
    )
    frame = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]
    observed = session.io.append(
        EventType.OBSERVATION, "environment", 0, {"frame": frame}
    )
    requested = session.io.append(EventType.TOOL_REQUEST, "tools", 0, {})
    request = ToolRequest(
        request_id="r1",
        tool_name="inspect_frame_region",
        arguments={
            "event_id": observed.event_id,
            "purpose": "zoom into the middle",
            "token_budget": 1024,
            "min_row": 0,
            "max_row": 1,
            "min_col": 1,
            "max_col": 2,
        },
    )
    result = execute_tool(
        session.io, request, requested, 0, FakeLineEnvironment().reset()
    )
    assert result.status == "complete"
    assert result.output["region"]["cells"] == [[1, 2], [5, 6]]
    assert result.evidence_refs == (observed.event_id,)


def test_inspect_frame_region_rejects_an_unknown_event_id(tmp_path) -> None:
    session = build_session(
        RunConfig(run_id="region-tool-404", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(),
        ScriptedModel([]),
    )
    requested = session.io.append(EventType.TOOL_REQUEST, "tools", 0, {})
    request = ToolRequest(
        request_id="r1",
        tool_name="inspect_frame_region",
        arguments={
            "event_id": "does-not-exist",
            "purpose": "zoom",
            "token_budget": 1024,
            "min_row": 0,
            "max_row": 1,
            "min_col": 0,
            "max_col": 1,
        },
    )
    try:
        execute_tool(session.io, request, requested, 0, FakeLineEnvironment().reset())
    except ValueError as error:
        assert "does-not-exist" in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown event id")
