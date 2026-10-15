"""PR-05 gate: an old negative experiment outcome is still queryable long
after it happened. Experiments already carry causal_refs to their
decision/action (`runtime/experiments.py`) and `retrieve_events`'s
EventType/causal_ref filters already serve this - a regression test for
reuse, not new machinery."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.retrieval import RetrievalQuery
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.evidence_tools import retrieve_events_tool
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script


def test_a_negative_experiment_outcome_remains_queryable_after_many_later_events(
    tmp_path: Path,
) -> None:
    config = RunConfig(run_id="ledger", experiment_id="test", output_dir=tmp_path)
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
            "action_event_id": action.event_id,
            "experiment": {
                "experiment_id": "e1",
                "question": "x",
                "hypothesis_ids": ["h1"],
            },
            "prediction_id": "p1",
            "has_checkable_prediction": True,
            "repeat": {"equivalent_prior": [], "justification": None},
            "unjustified_repeat": False,
            "verification_status": "mismatch",
        },
        (decision.event_id, action.event_id),
    )
    for step in range(1, 40):
        io.append(EventType.OBSERVATION, "environment", step, {"noise": step})

    query = RetrievalQuery(event_types=(EventType.EXPERIMENT,))
    result = retrieve_events_tool(
        io,
        ToolRequest(
            request_id="q",
            tool_name="retrieve_events",
            arguments=query.model_dump(mode="json"),
        ),
    )
    matches = result.output["retrieval"]["matches"]
    assert len(matches) == 1
    assert matches[0]["event_id"] == negative.event_id
    assert matches[0]["payload"]["verification_status"] == "mismatch"

    by_causal_ref = RetrievalQuery(causal_ref=action.event_id)
    result2 = retrieve_events_tool(
        io,
        ToolRequest(
            request_id="q2",
            tool_name="retrieve_events",
            arguments=by_causal_ref.model_dump(mode="json"),
        ),
    )
    assert negative.event_id in [
        m["event_id"] for m in result2.output["retrieval"]["matches"]
    ]
