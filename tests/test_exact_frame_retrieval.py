"""The exact frame stays retrievable by ID and crops exactly on request."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig, BeliefConfig
from arc_agi_3.context.evidence_retrieval import retrieve_evidence
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.evidence import EvidenceQuery, FrameRegion
from arc_agi_3.runtime.context import project_context
from arc_agi_3.trace.store import JsonlEventStore
from tests.test_compact_observation import big_observation


def store(tmp_path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        tmp_path / "events.jsonl",
        "run",
        "episode",
        "main",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_current_observation_is_retrievable_exactly_by_id(tmp_path: Path) -> None:
    events = store(tmp_path)
    observation = big_observation()
    events.append(
        EventType.OBSERVATION, "environment", 0, observation.model_dump(mode="json")
    )
    context = project_context(
        0,
        observation,
        BudgetLedger({}),
        events,
        CognitiveWorkspace(BeliefConfig()),
        AblationConfig(),
    )
    assert context.current_observation_event_id is not None
    status, output, refs = retrieve_evidence(
        events.read(),
        EvidenceQuery(
            event_ids=(context.current_observation_event_id,),
            view="full",
            purpose="inspect the exact grid",
            token_budget=16_384,
        ),
    )
    assert status == "complete"
    assert (
        output["items"][0]["content"]["frame"]
        == observation.model_dump(mode="json")["frame"]
    )
    assert refs == (context.current_observation_event_id,)


def test_region_inspection_returns_the_exact_requested_crop(tmp_path: Path) -> None:
    events = store(tmp_path)
    observation = big_observation()
    event = events.append(
        EventType.OBSERVATION, "environment", 0, observation.model_dump(mode="json")
    )
    query = EvidenceQuery(
        event_ids=(event.event_id,),
        view="region",
        purpose="zoom into the top-left corner",
        token_budget=4096,
        region=FrameRegion(layer=0, x=0, y=0, width=3, height=2),
    )
    _, output, _ = retrieve_evidence(events.read(), query)
    cells = output["items"][0]["content"]["cells"]
    assert cells == [list(row[:3]) for row in observation.frame[0][:2]]
