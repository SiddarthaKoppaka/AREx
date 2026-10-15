"""PR-07 gate: a batch of independent tool_requests in one decision stays
deterministically ordered, isolates partial failures (one failing call
never aborts the rest), has no missing/duplicate event refs, and produces
identical results across repeated runs. Most of this already holds by
construction (`runtime/cognition.py`'s sequential loop, `run_tool`'s own
per-call try/except, `check_tool_requests`'s request_id uniqueness) - this
is the regression suite proving it, plus the new typed batch summary."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.retrieval import RetrievalQuery
from arc_agi_3.evidence import summarize_tool_batch
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.tools import run_tool
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script

MISSING_REF = "artifacts/" + "0" * 64 + ".json"


def _run_batch(tmp_path: Path) -> tuple:
    config = RunConfig(run_id="batch", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    io = runner.io
    observation = io.environment.reset()
    decision = io.append(EventType.MODEL_DECISION, "model", 0, {"decision": {}})
    requests = (
        ToolRequest(
            request_id="ok-1",
            tool_name="retrieve_events",
            arguments=RetrievalQuery(event_types=(EventType.OBSERVATION,)).model_dump(
                mode="json"
            ),
        ),
        ToolRequest(
            request_id="missing",
            tool_name="retrieve_artifact",
            arguments={
                "artifact_ref": MISSING_REF,
                "purpose": "x",
                "token_budget": 512,
            },
        ),
        ToolRequest(
            request_id="ok-2",
            tool_name="retrieve_events",
            arguments=RetrievalQuery(
                event_types=(EventType.MODEL_DECISION,)
            ).model_dump(mode="json"),
        ),
    )
    for request in requests:
        run_tool(io, request, decision, 0, observation)
    return io, decision


def test_partial_failure_is_isolated_and_order_is_deterministic(tmp_path: Path) -> None:
    io, decision = _run_batch(tmp_path)
    summary = summarize_tool_batch(io.events.read(), decision.event_id)
    assert summary.requested == 3
    assert summary.succeeded == 2
    assert summary.failed == 1
    assert summary.request_ids_in_order == ("ok-1", "missing", "ok-2")


def test_no_missing_or_duplicate_event_refs(tmp_path: Path) -> None:
    io, decision = _run_batch(tmp_path)
    events = io.events.read()
    requested = [
        e
        for e in events
        if e.event_type is EventType.TOOL_REQUEST and decision.event_id in e.causal_refs
    ]
    results = [e for e in events if e.event_type is EventType.TOOL_RESULT]
    assert len(requested) == len(results) == 3
    result_sources = [e.causal_refs[0] for e in results]
    assert len(result_sources) == len(set(result_sources))
    assert set(result_sources) == {e.event_id for e in requested}


def test_the_same_batch_produces_identical_results_across_runs(tmp_path: Path) -> None:
    io1, decision1 = _run_batch(tmp_path / "a")
    io2, decision2 = _run_batch(tmp_path / "b")
    summary1 = summarize_tool_batch(io1.events.read(), decision1.event_id)
    summary2 = summarize_tool_batch(io2.events.read(), decision2.event_id)
    assert summary1.model_dump(exclude={"decision_event_id"}) == summary2.model_dump(
        exclude={"decision_event_id"}
    )
