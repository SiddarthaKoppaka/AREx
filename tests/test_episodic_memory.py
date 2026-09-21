"""Older events form bounded episodic memory without losing exact history."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig, BeliefConfig
from arc_agi_3.context import EventRetriever
from arc_agi_3.context.episodic import summarize_episode
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.retrieval import RetrievalQuery
from arc_agi_3.runtime.context import project_context
from arc_agi_3.trace.store import JsonlEventStore


def observation(value: int) -> Observation:
    return Observation.build(
        game_id="memory",
        frame=[[[value] * 4 for _ in range(4)]],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=[1],
    )


def event_store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def project(events: JsonlEventStore, persistent: bool = True):
    return project_context(
        12,
        observation(99),
        BudgetLedger({}),
        events,
        CognitiveWorkspace(BeliefConfig()),
        AblationConfig(persistent_memory=persistent),
        recent_event_limit=4,
        episodic_retrieval_limit=4,
    )


def test_older_history_is_deterministically_summarized(tmp_path: Path) -> None:
    events = event_store(tmp_path / "events.jsonl")
    first_id = ""
    for step in range(12):
        observed = events.append(
            EventType.OBSERVATION,
            "environment",
            step,
            observation(step).model_dump(mode="json"),
        )
        first_id = first_id or observed.event_id
        events.append(EventType.ACTION, "executor", step, {"action": {"action_id": 1}})
    exact_before = events.read()
    first, second = project(events), project(events)
    assert first.working_scratchpad is not None
    assert first.episodic_memory == second.episodic_memory
    memory = first.episodic_memory
    assert memory is not None
    assert [item.step_id for item in memory.items] == [6, 7, 8, 9]
    assert memory.source_event_count == 20
    assert memory.summarized_event_count == 8
    assert memory.omitted_event_count == 12
    assert memory.items[0].summary == "Executed action 1."
    assert memory.items[0].facts_learned[0].startswith("Observed state")
    assert memory.items[0].evidence_refs == memory.items[0].source_event_refs
    expected_recent = tuple(event.event_id for event in exact_before[-4:])
    assert first.recent_event_refs == expected_recent
    assert all(
        "frame" not in str(event.payload)
        for item in memory.items
        for event in item.events
    )
    recent_refs = set(first.recent_event_refs)
    assert all(
        not recent_refs.intersection(item.source_event_refs) for item in memory.items
    )
    assert events.read() == exact_before
    result = EventRetriever(events.read()).retrieve(
        RetrievalQuery(event_ids=(first_id,))
    )
    assert "frame" in result.matches[0].payload


def test_persistent_memory_ablation_removes_episode_summary(tmp_path: Path) -> None:
    events = event_store(tmp_path / "events.jsonl")
    for step in range(3):
        events.append(
            EventType.OBSERVATION,
            "environment",
            step,
            observation(step).model_dump(mode="json"),
        )
    assert project(events, persistent=False).episodic_memory is None


def test_episode_retrieval_prefers_objective_terms_over_recency(tmp_path: Path) -> None:
    events = event_store(tmp_path / "relevance.jsonl")
    events.append(
        EventType.RECOVERY,
        "recovery",
        1,
        {"status": "failed", "reason": "restore_unsupported"},
    )
    for step in range(2, 6):
        events.append(EventType.ACTION, "executor", step, {"action": {"action_id": 1}})
    memory = summarize_episode(events.read(), item_limit=1, query="restore checkpoint")
    assert memory is not None
    assert memory.items[0].step_id == 1
    assert memory.items[0].failed_approaches == (
        "Recovery failed: restore_unsupported.",
    )
