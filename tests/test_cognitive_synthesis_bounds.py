"""COGNITIVE_SYNTHESIS stays bounded with an explicit omitted count instead
of growing with every specialist report/review for the whole episode.
See PR-02-scoping.md sections 2 and 4-B."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.contracts.cognitive_faculty import CognitiveFaculty
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.contracts.specialist_review import SpecialistReview
from arc_agi_3.evidence.cognitive_synthesis import build_cognitive_synthesis
from arc_agi_3.trace.store import JsonlEventStore


def store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def report(turn: int) -> SpecialistReport:
    return SpecialistReport(
        specialist_id=f"s{turn}",
        faculty=CognitiveFaculty.HYPOTHESIS,
        turn=turn,
        assessment=f"report at turn {turn}",
    )


def review(turn: int) -> SpecialistReview:
    return SpecialistReview(
        specialist_id=f"r{turn}", faculty=CognitiveFaculty.CRITIC, turn=turn
    )


def test_synthesis_keeps_the_most_recent_reports_with_an_omitted_count(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    for turn in range(12):
        events.append(
            EventType.SPECIALIST_REPORT,
            "exocortex",
            turn,
            {"report": report(turn).model_dump(mode="json")},
        )
    synthesis = build_cognitive_synthesis(events.read())
    assert synthesis is not None
    assert len(synthesis.reports) == 8
    assert synthesis.omitted_report_count == 4
    assert [r.turn for r in synthesis.reports] == list(range(4, 12))


def test_synthesis_bounds_reviews_independently_of_reports(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    events.append(
        EventType.SPECIALIST_REPORT,
        "exocortex",
        0,
        {"report": report(0).model_dump(mode="json")},
    )
    for turn in range(10):
        events.append(
            EventType.SPECIALIST_REVIEW,
            "exocortex",
            turn,
            {"review": review(turn).model_dump(mode="json")},
        )
    synthesis = build_cognitive_synthesis(events.read())
    assert synthesis is not None
    assert len(synthesis.reviews) == 8
    assert synthesis.omitted_review_count == 2
