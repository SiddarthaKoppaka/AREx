"""PR-05 gate: a pure exact repeat avoids recompute while staying fully
observable in the trace, a changed argument bypasses the cache, and a
write tool is never cached even on an exact repeat.

`retrieve_evidence`/`inspect_frame_region` were added to `CACHEABLE_TOOLS`
after re-verifying (see `evidence/tool_cache.py`'s docstring) that both
target one fixed, already-existing event ID whose relevant history is
bounded by that event's own sequence number - the same exact-repeat gate
applies to them as to `retrieve_artifact`."""

from pathlib import Path

from arc_agi_3.contracts.artifacts import ArchiveArtifactRequest
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.evidence import EvidenceQuery, FrameRegionRequest
from tests.tool_result_cache_fixtures import archive, payload, read_request, setup


def test_an_exact_repeat_is_a_cache_hit_and_stays_observable(tmp_path: Path) -> None:
    io, decision, observation = setup(tmp_path)
    ref = archive(io, decision, observation, k="v")
    request = read_request(ref, "r2")
    first = payload(io, request, decision, observation)
    second = payload(io, request, decision, observation)
    assert "cache_hit" not in first
    assert second["cache_hit"] is True
    assert second["cached_from_event_id"]
    assert second["result"]["output"] == first["result"]["output"]


def test_a_different_argument_bypasses_the_cache(tmp_path: Path) -> None:
    io, decision, observation = setup(tmp_path)
    ref = archive(io, decision, observation, k="v", k2="w")
    payload(io, read_request(ref, "r2"), decision, observation)
    second = payload(io, read_request(ref, "r3", keys=("k",)), decision, observation)
    assert "cache_hit" not in second


def test_retrieve_evidence_on_a_fixed_event_is_a_cache_hit(tmp_path: Path) -> None:
    io, decision, observation = setup(tmp_path)
    obs_event = io.append(
        EventType.OBSERVATION, "environment", 0, {"frame": [[[1, 2], [3, 4]]]}
    )
    request = ToolRequest(
        request_id="r1",
        tool_name="retrieve_evidence",
        arguments=EvidenceQuery(
            event_ids=(obs_event.event_id,),
            view="rle_frame",
            purpose="inspect",
            token_budget=1024,
        ).model_dump(mode="json"),
    )
    first = payload(io, request, decision, observation)
    second = payload(io, request, decision, observation)
    assert "cache_hit" not in first
    assert second["cache_hit"] is True
    assert second["result"]["output"] == first["result"]["output"]


def test_inspect_frame_region_on_a_fixed_event_is_a_cache_hit(tmp_path: Path) -> None:
    io, decision, observation = setup(tmp_path)
    obs_event = io.append(
        EventType.OBSERVATION, "environment", 0, {"frame": [[[1, 2], [3, 4]]]}
    )
    request = ToolRequest(
        request_id="r1",
        tool_name="inspect_frame_region",
        arguments=FrameRegionRequest(
            event_id=obs_event.event_id,
            purpose="zoom",
            token_budget=1024,
            min_row=0,
            max_row=0,
            min_col=0,
            max_col=1,
        ).model_dump(mode="json"),
    )
    first = payload(io, request, decision, observation)
    second = payload(io, request, decision, observation)
    assert "cache_hit" not in first
    assert second["cache_hit"] is True
    assert second["result"]["output"] == first["result"]["output"]


def test_a_write_tool_is_never_cached_even_on_an_exact_repeat(tmp_path: Path) -> None:
    io, decision, observation = setup(tmp_path)
    request = ToolRequest(
        request_id="r1",
        tool_name="archive_artifact",
        arguments=ArchiveArtifactRequest(
            purpose="keep", evidence_refs=(decision.event_id,), content={"k": "v"}
        ).model_dump(mode="json"),
    )
    first = payload(io, request, decision, observation)
    second = payload(io, request, decision, observation)
    assert "cache_hit" not in first
    assert "cache_hit" not in second
