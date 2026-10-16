"""PR-12: `context_stats()` (runtime/context_stats.py) wires the two new
live counters through `project_context` end-to-end, not just as pure
functions - see `tests/test_live_novelty_metrics.py` for those."""

from pathlib import Path

from arc_agi_3.contracts.cognition import ToolRequest
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from tests.epistemic_policies import long_run


def test_tool_requests_since_last_action_is_wired_live_into_context_stats(
    tmp_path: Path,
) -> None:
    investigate = CognitiveDecision(
        assessment="x",
        intent="x",
        mode=DecisionMode.INVESTIGATE,
        tool_requests=(
            ToolRequest(
                request_id="r1",
                tool_name="retrieve_evidence",
                arguments={
                    "event_ids": ["obs-1"],
                    "view": "rle_frame",
                    "purpose": "inspect",
                    "token_budget": 1024,
                },
            ),
        ),
    )
    _, model = long_run(tmp_path, [lambda _: investigate])
    late = model.contexts[-1]
    assert late.context_stats.tool_requests_since_last_action == 1
    assert isinstance(late.context_stats.untried_actions_from_current_state, tuple)
