"""Bounded working set: prompt stops growing, exact evidence stays retrievable."""

from pathlib import Path

from arc_agi_3.adapters.prompts import decision_prompt
from arc_agi_3.context.evidence_retrieval import retrieve_evidence
from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.evidence import EvidenceQuery
from tests.epistemic_policies import TURNS, long_run


def test_prompt_is_bounded_and_keeps_live_evidence(tmp_path: Path) -> None:
    runner, model = long_run(tmp_path, [])
    sizes = [len(decision_prompt(context)) for context in model.contexts]
    assert max(sizes[6:]) < 1.15 * sizes[5]
    late = model.contexts[-1]
    assert late.hypothesis_ledger[0].hypothesis.hypothesis_id == "h1"
    assert late.hypothesis_ledger[0].omitted_tests == TURNS - 3
    assert late.latest_transition is not None and late.latest_verification is not None
    assert late.context_stats is not None
    assert late.context_stats.episodic_items_included <= 6
    assert "c" in late.context_stats.episodic_query
    assert all(e.event_type is not EventType.BUDGET for e in late.recent_events)
    decision = [
        e for e in runner.events.read() if e.event_type is EventType.MODEL_DECISION
    ]
    assert decision[-1].payload["context"]["history_events"] > 100


def test_exact_older_evidence_remains_retrievable(tmp_path: Path) -> None:
    runner, model = long_run(tmp_path, [])
    events = runner.events.read()
    first = next(e for e in events if e.event_type is EventType.TRANSITION)
    late = model.contexts[-1]
    assert first.event_id not in late.recent_event_refs
    query = EvidenceQuery(
        event_ids=(first.event_id,),
        view="transition_delta",
        purpose="re-examine first move",
        token_budget=4096,
    )
    _, output, refs = retrieve_evidence(events, query)
    assert refs == (first.event_id,)
    assert output["items"][0]["content"]["changes"] == [
        [0, 2, 1, 3, 0],
        [0, 2, 3, 0, 3],
    ]


def test_retrodiction_reports_compatibility_without_inferring_truth(
    tmp_path: Path,
) -> None:
    request = ToolRequest(
        request_id="retro",
        tool_name="check_prediction_history",
        arguments={"prediction": {"no_op": True}, "action": {"action_id": 1}},
    )
    investigate = CognitiveDecision(
        assessment="Check history.",
        intent="Test a prediction retrospectively.",
        mode=DecisionMode.INVESTIGATE,
        tool_requests=(request,),
    )
    runner, model = long_run(tmp_path, [lambda _: investigate])
    result = next(
        e for e in runner.events.read() if e.event_type is EventType.TOOL_RESULT
    ).payload["result"]
    retro = result["output"]["retrodiction"]
    # Seven ACTION1 transitions: five shifts, then two no-ops at the grid edge.
    assert (len(retro["contradicting"]), len(retro["compatible"])) == (5, 2)
    assert retro["examined"] == TURNS // 2 and result["status"] == "complete"
    visible = [
        e
        for e in model.contexts[-1].recent_events
        if e.event_type.value == "tool_result"
    ]
    assert visible[0].payload["result"]["retrodiction"]["examined"] == TURNS // 2
    assert runner.io.workspace.beliefs.current[0].probability == 0.5
