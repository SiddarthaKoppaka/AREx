"""PR-08 gate: an old negative experiment is recovered after 50+ unrelated
turns, through both the live retrieve_events tool and the offline
TraceIndex (SQLite FTS5 - no embeddings, no vector dependency, per the
explicit constraint). A second trace from a different game never leaks
into the first's results. Measured directly (exact counts), not assumed."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.retrieval import RetrievalQuery
from arc_agi_3.research import TraceIndex
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.evidence_tools import retrieve_events_tool
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script

UNRELATED_TURNS = 55


def _build_trace(tmp_path: Path, run_id: str, game_id: str) -> tuple:
    config = RunConfig(run_id=run_id, experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(), ScriptedModel(successful_script())
    )
    io = runner.io
    decision = io.append(EventType.MODEL_DECISION, "model", 0, {"decision": {}})
    action = io.append(EventType.ACTION, "executor", 0, {"action": {"action_id": 1}})
    negative = io.append(
        EventType.EXPERIMENT,
        "verifier",
        0,
        {
            "game_id": game_id,
            "verification_status": "mismatch",
            "question": f"{game_id}-q1",
        },
        (decision.event_id, action.event_id),
    )
    for step in range(1, UNRELATED_TURNS):
        io.append(
            EventType.OBSERVATION,
            "environment",
            step,
            {"noise": step, "game_id": game_id},
        )
    return runner, negative


def test_retrieve_events_recovers_an_old_negative_experiment_after_many_turns(
    tmp_path: Path,
) -> None:
    runner, negative = _build_trace(tmp_path, "long", "game-a")
    query = RetrievalQuery(event_types=(EventType.EXPERIMENT,))
    result = retrieve_events_tool(
        runner.io,
        ToolRequest(
            request_id="q",
            tool_name="retrieve_events",
            arguments=query.model_dump(mode="json"),
        ),
    )
    matches = result.output["retrieval"]["matches"]
    assert len(matches) == 1
    assert matches[0]["event_id"] == negative.event_id


def test_trace_index_also_recovers_the_same_old_negative_experiment(
    tmp_path: Path,
) -> None:
    runner, negative = _build_trace(tmp_path, "long", "game-a")
    index = TraceIndex(tmp_path / "derived.sqlite3")
    count = index.rebuild(runner.events.path)
    assert count == UNRELATED_TURNS + 2
    hits = index.search("mismatch")
    assert any(event_id == negative.event_id for event_id, _ in hits)


def test_a_second_games_trace_never_leaks_into_the_first_games_results(
    tmp_path: Path,
) -> None:
    runner_a, negative_a = _build_trace(tmp_path / "a", "long", "game-a")
    runner_b, negative_b = _build_trace(tmp_path / "b", "long", "game-b")
    query = RetrievalQuery(event_types=(EventType.EXPERIMENT,))
    result_a = retrieve_events_tool(
        runner_a.io,
        ToolRequest(
            request_id="qa",
            tool_name="retrieve_events",
            arguments=query.model_dump(mode="json"),
        ),
    )
    matches_a = {m["event_id"] for m in result_a.output["retrieval"]["matches"]}
    assert matches_a == {negative_a.event_id}
    assert negative_b.event_id not in matches_a
