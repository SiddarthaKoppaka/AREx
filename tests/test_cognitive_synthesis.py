"""build_cognitive_synthesis aggregates mechanically from the full trace."""

from datetime import UTC, datetime
from pathlib import Path

from arc_agi_3.contracts.enums import CognitiveFaculty, EventType
from arc_agi_3.contracts.specialist_report import CandidateHypothesis, SpecialistReport
from arc_agi_3.contracts.specialist_review import SpecialistCritique, SpecialistReview
from arc_agi_3.evidence import build_cognitive_synthesis
from arc_agi_3.trace.store import JsonlEventStore


def store(path: Path) -> JsonlEventStore:
    return JsonlEventStore(
        path, "run", "episode", "main", lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_build_cognitive_synthesis_flattens_mechanically_without_interpreting(
    tmp_path: Path,
) -> None:
    events = store(tmp_path / "events.jsonl")
    hypothesis_report = SpecialistReport(
        specialist_id="S-hypothesis",
        faculty=CognitiveFaculty.HYPOTHESIS,
        turn=1,
        assessment="x",
        hypotheses=(CandidateHypothesis(claim="11 is a goal", confidence=0.7),),
    )
    critic_review = SpecialistReview(
        specialist_id="S-critic",
        faculty=CognitiveFaculty.CRITIC,
        turn=1,
        critiques=(
            SpecialistCritique(
                target_specialist_id="S-hypothesis",
                target_claim="11 is a goal",
                critique="unsupported",
            ),
        ),
    )
    events.append(
        EventType.SPECIALIST_REPORT,
        "exocortex",
        1,
        {"report": hypothesis_report.model_dump(mode="json")},
    )
    events.append(
        EventType.SPECIALIST_REVIEW,
        "exocortex",
        1,
        {"review": critic_review.model_dump(mode="json")},
    )
    synthesis = build_cognitive_synthesis(events.read())
    assert synthesis is not None
    assert synthesis.candidate_hypotheses[0].claim == "11 is a goal"
    assert synthesis.candidate_hypotheses[0].specialist_id == "S-hypothesis"
    assert synthesis.flagged_contradictions == (
        "S-critic on S-hypothesis: unsupported",
    )


def test_build_cognitive_synthesis_is_none_with_no_reports(tmp_path: Path) -> None:
    events = store(tmp_path / "events.jsonl")
    assert build_cognitive_synthesis(events.read()) is None
