"""Shared setup for `tests/test_tool_result_cache.py`: builds a real
`EpisodeIO` via `build_runner` and runs tools through `run_tool` directly,
rather than threading values through a full decide loop."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.artifacts import ArchiveArtifactRequest, ReadArtifactRequest
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.tools import run_tool
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    config = RunConfig(run_id="cache", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    observation = runner.io.environment.reset()
    decision = runner.io.append(EventType.MODEL_DECISION, "model", 0, {"decision": {}})
    return runner.io, decision, observation


def payload(io, request, decision, observation) -> dict:  # type: ignore[no-untyped-def]
    run_tool(io, request, decision, 0, observation)
    return io.events.read()[-1].payload


def archive(io, decision, observation, **content: object) -> str:  # type: ignore[no-untyped-def]
    request = ToolRequest(
        request_id="archive",
        tool_name="archive_artifact",
        arguments=ArchiveArtifactRequest(
            purpose="keep", evidence_refs=(decision.event_id,), content=content
        ).model_dump(mode="json"),
    )
    result = payload(io, request, decision, observation)["result"]
    return result["output"]["artifact_ref"]  # type: ignore[no-any-return]


def read_request(ref: str, request_id: str, **extra: object) -> ToolRequest:
    return ToolRequest(
        request_id=request_id,
        tool_name="retrieve_artifact",
        arguments=ReadArtifactRequest(
            artifact_ref=ref, purpose="read", token_budget=1024, **extra
        ).model_dump(mode="json"),
    )
