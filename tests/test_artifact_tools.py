"""LM-authored artifacts can be reopened locally under a strict budget."""

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime.artifact_tools import archive_artifact, retrieve_artifact
from arc_agi_3.runtime.factory import build_session
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def test_archive_and_retrieve_artifact_by_reference(tmp_path) -> None:
    session = build_session(
        RunConfig(run_id="artifacts", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(),
        ScriptedModel([]),
    )
    evidence = session.io.append(EventType.OBSERVATION, "environment", 0, {"value": 1})
    requested = session.io.append(EventType.TOOL_REQUEST, "tools", 0, {})
    archived = archive_artifact(
        session.io,
        ToolRequest(
            request_id="archive",
            tool_name="archive_artifact",
            arguments={
                "purpose": "Keep a compact phase handoff",
                "evidence_refs": [evidence.event_id],
                "content": {"action_model": {"1": "moves"}},
            },
        ),
        requested,
    )
    reference = archived.output["artifact_ref"]
    assert isinstance(reference, str)
    reopened = retrieve_artifact(
        session.io,
        ToolRequest(
            request_id="read",
            tool_name="retrieve_artifact",
            arguments={
                "artifact_ref": reference,
                "purpose": "Resume action learning",
                "token_budget": 2048,
                "keys": ["content"],
            },
        ),
    )
    assert reopened.status == "complete"
    assert reopened.output["content"] == {"content": {"action_model": {"1": "moves"}}}


def test_artifact_read_fails_closed_for_missing_reference(tmp_path) -> None:
    session = build_session(
        RunConfig(run_id="missing", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(),
        ScriptedModel([]),
    )
    request = ToolRequest(
        request_id="read",
        tool_name="retrieve_artifact",
        arguments={
            "artifact_ref": f"artifacts/{'0' * 64}.json",
            "purpose": "Inspect missing evidence",
            "token_budget": 512,
        },
    )
    try:
        retrieve_artifact(session.io, request)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing artifact must fail visibly")


def test_large_context_reference_always_exists(tmp_path) -> None:
    session = build_session(
        RunConfig(run_id="projected", experiment_id="test", output_dir=tmp_path),
        FakeLineEnvironment(),
        ScriptedModel([]),
    )
    session.io.append(EventType.TOOL_RESULT, "tools", 0, {"blob": "x" * 5000})
    context = session.io.context(0, session.io.environment.reset())
    reference = context.recent_events[0].payload["artifact_ref"]
    assert isinstance(reference, str)
    assert session.io.events.artifacts.read(reference) == {"blob": "x" * 5000}
