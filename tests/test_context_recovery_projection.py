"""Bounded recent and recovery context projections."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig, BeliefConfig
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.recovery import reconstruct_recovery
from arc_agi_3.runtime.context import project_context
from arc_agi_3.trace.store import JsonlEventStore


def observation(value: int) -> Observation:
    return Observation.build(
        game_id="large",
        frame=[[[value] * 64 for _ in range(64)]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1],
    )


def store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_recent_window_and_nested_tool_output_are_bounded(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    current = observation(1)
    events.append(
        EventType.OBSERVATION, "environment", 0, current.model_dump(mode="json")
    )
    for step in range(10):
        events.append(
            EventType.TOOL_RESULT,
            "tools",
            step,
            {
                "result": {
                    "status": "complete",
                    "output": {
                        "retrieval": {
                            "matches": [
                                {"payload": {"frame": [[list(range(64))] * 64]}}
                            ]
                        }
                    },
                }
            },
        )
    first = project_context(
        1,
        current,
        BudgetLedger({}),
        events,
        CognitiveWorkspace(BeliefConfig()),
        AblationConfig(),
        3,
    )
    second = project_context(
        1,
        current,
        BudgetLedger({}),
        events,
        CognitiveWorkspace(BeliefConfig()),
        AblationConfig(),
        3,
    )
    assert first == second
    assert len(first.recent_events) == 3
    assert all("frame" not in str(item.payload) for item in first.recent_events)
    assert all(
        item.payload["result"]["status"] == "complete" for item in first.recent_events
    )


def test_recovery_evidence_never_embeds_historical_frame(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    old = observation(7)
    events.append(EventType.OBSERVATION, "environment", 0, old.model_dump(mode="json"))
    events.append(EventType.ACTION, "executor", 1, {"action": {"action_id": 1}})
    events.append(
        EventType.VERIFICATION,
        "verifier",
        1,
        {"checked": True, "passed": False, "mismatches": ["state"]},
    )
    evidence = reconstruct_recovery(events.read())
    assert evidence is not None
    assert evidence.local_events[0].payload["observation_hash"] == old.observation_hash
    assert all("frame" not in str(item.payload) for item in evidence.local_events)
