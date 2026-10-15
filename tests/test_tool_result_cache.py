"""PR-05 gate: a pure exact repeat avoids recompute while staying fully
observable in the trace, a changed argument bypasses the cache, and a
write tool is never cached even on an exact repeat. Calls `run_tool`
directly against a real `EpisodeIO` (from `build_runner`) rather than
threading values through a full decide loop."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.artifacts import ArchiveArtifactRequest, ReadArtifactRequest
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.tools import run_tool
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def _setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    config = RunConfig(run_id="cache", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    observation = runner.io.environment.reset()
    decision = runner.io.append(EventType.MODEL_DECISION, "model", 0, {"decision": {}})
    return runner.io, decision, observation


def _payload(io, request, decision, observation) -> dict:  # type: ignore[no-untyped-def]
    run_tool(io, request, decision, 0, observation)
    return io.events.read()[-1].payload


def _archive(io, decision, observation, **content: object) -> str:  # type: ignore[no-untyped-def]
    request = ToolRequest(
        request_id="archive",
        tool_name="archive_artifact",
        arguments=ArchiveArtifactRequest(
            purpose="keep", evidence_refs=(decision.event_id,), content=content
        ).model_dump(mode="json"),
    )
    result = _payload(io, request, decision, observation)["result"]
    return result["output"]["artifact_ref"]  # type: ignore[no-any-return]


def _read(ref: str, request_id: str, **extra: object) -> ToolRequest:
    return ToolRequest(
        request_id=request_id,
        tool_name="retrieve_artifact",
        arguments=ReadArtifactRequest(
            artifact_ref=ref, purpose="read", token_budget=1024, **extra
        ).model_dump(mode="json"),
    )


def test_an_exact_repeat_is_a_cache_hit_and_stays_observable(tmp_path: Path) -> None:
    io, decision, observation = _setup(tmp_path)
    ref = _archive(io, decision, observation, k="v")
    read_request = _read(ref, "r2")
    first = _payload(io, read_request, decision, observation)
    second = _payload(io, read_request, decision, observation)
    assert "cache_hit" not in first
    assert second["cache_hit"] is True
    assert second["cached_from_event_id"]
    assert second["result"]["output"] == first["result"]["output"]


def test_a_different_argument_bypasses_the_cache(tmp_path: Path) -> None:
    io, decision, observation = _setup(tmp_path)
    ref = _archive(io, decision, observation, k="v", k2="w")
    _payload(io, _read(ref, "r2"), decision, observation)
    second = _payload(io, _read(ref, "r3", keys=("k",)), decision, observation)
    assert "cache_hit" not in second


def test_a_write_tool_is_never_cached_even_on_an_exact_repeat(tmp_path: Path) -> None:
    io, decision, observation = _setup(tmp_path)
    request = ToolRequest(
        request_id="r1",
        tool_name="archive_artifact",
        arguments=ArchiveArtifactRequest(
            purpose="keep", evidence_refs=(decision.event_id,), content={"k": "v"}
        ).model_dump(mode="json"),
    )
    first = _payload(io, request, decision, observation)
    second = _payload(io, request, decision, observation)
    assert "cache_hit" not in first
    assert "cache_hit" not in second
