"""PR-04 gate: replay never writes a predicted state back to the trace as
if it were a real observation - it only reads existing cases."""

from pathlib import Path

from arc_agi_3.evidence.history import transition_records
from arc_agi_3.executable_model import build_transition_cases, replay_candidate
from tests.executable_model_fixtures import (
    CORRECT_SOURCE,
    candidate,
    record_shift_trace,
)


def test_replay_never_appends_to_the_trace_or_changes_its_content(
    tmp_path: Path,
) -> None:
    events = record_shift_trace(tmp_path, turns=4)
    trace_path = tmp_path / "replay" / "events.jsonl"
    before_text = trace_path.read_text()
    cases = build_transition_cases(
        transition_records(events), seed=0, holdout_fraction=0.5
    )
    report = replay_candidate(candidate(CORRECT_SOURCE), CORRECT_SOURCE, cases)
    assert report.total_cases == len(cases)
    assert trace_path.read_text() == before_text
