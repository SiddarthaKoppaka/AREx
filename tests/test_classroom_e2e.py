"""A Skeptic contests a Scientist's claim; the Teacher's action is still the
only thing with authority, and the audit confirms it."""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment
from tests.classroom_e2e_fixtures import one_shot_teacher, student_outputs
from tests.generation_fixtures import ScriptedBackend


def test_skeptic_contradiction_is_flagged_and_boundary_audit_still_passes(
    tmp_path: Path,
) -> None:
    students = ScriptedBackend(student_outputs())
    teacher = one_shot_teacher()
    config = RunConfig(
        run_id="classroom-e2e", experiment_id="test", output_dir=tmp_path
    )
    runner = build_runner(
        config,
        FakeLineEnvironment(target=1),
        teacher,
        students=students,
        raise_on_failure=True,
    )
    result = runner.run()

    assert result.stop_reason == "terminal"
    assert result.metrics.succeeded is True
    assert result.metrics.student_reports_generated == 3
    assert result.metrics.peer_reviews_generated == 3
    assert result.metrics.flagged_contradictions_raised == 1

    events = runner.events.read()
    synthesis = next(e for e in events if e.event_type is EventType.CLASSROOM_SYNTHESIS)
    assert synthesis.payload["candidate_hypotheses"][0]["claim"] == "value 11 is a goal"
    assert synthesis.payload["flagged_contradictions"] == [
        "S-skeptic on S-scientist: evidence only proves cells changed, not goal status"
    ]
    assert audit_agency_boundary(events) == ()
