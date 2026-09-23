"""Automatic context stays small without mutating authoritative events."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.adapters.prompts import decision_prompt
from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig, BeliefConfig
from arc_agi_3.context import EventRetriever
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.retrieval import RetrievalQuery
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


def test_current_frame_occurs_once_and_history_is_compacted(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    old, current = observation(2), observation(3)
    first = events.append(
        EventType.OBSERVATION, "environment", 0, old.model_dump(mode="json")
    )
    events.append(
        EventType.ACTION,
        "executor",
        1,
        {"action": {"action_id": 1}, "before_hash": old.observation_hash},
    )
    events.append(
        EventType.OBSERVATION, "environment", 1, current.model_dump(mode="json")
    )
    context = project_context(
        1,
        current,
        BudgetLedger({}),
        events,
        CognitiveWorkspace(BeliefConfig()),
        AblationConfig(),
    )
    prompt = decision_prompt(context)
    assert '"frame":' not in prompt
    assert '"encoding":"row_rle_v1"' in prompt
    assert "[[3,64]]" in prompt
    assert '"frame":[[[2' not in prompt
    assert context.recent_events[0].payload["observation_hash"] == old.observation_hash
    assert context.recent_events[1].payload["action"] == {"action_id": 1}
    assert context.recent_event_refs[0] == first.event_id
    assert events.read()[0].payload["frame"] == old.model_dump(mode="json")["frame"]
    exact = EventRetriever(events.read()).retrieve(
        RetrievalQuery(event_ids=(first.event_id,))
    )
    assert exact.matches[0].payload["frame"] == old.model_dump(mode="json")["frame"]
    assert len(prompt) < 32_768


def test_recent_projection_keeps_complete_latest_turn(tmp_path: Path) -> None:
    events = store(tmp_path / "turns.jsonl")
    for amount in range(5):
        events.append(EventType.BUDGET, "budget", 4, {"amount": amount})
    context = project_context(
        4,
        observation(1),
        BudgetLedger({}),
        events,
        CognitiveWorkspace(BeliefConfig()),
        AblationConfig(),
        recent_event_limit=2,
        raw_recent_turns=1,
    )
    assert len(context.recent_events) == 5
    assert context.episodic_memory is None
