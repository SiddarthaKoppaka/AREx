"""The PR-04 gate: a deliberately wrong model is falsified on holdout
cases, and a revised candidate (correct parent lineage) improves held-out
accuracy. See tests/test_executable_model_trace_integrity.py for the
"no synthetic result written as a real observation" gate item."""

from pathlib import Path

from arc_agi_3.executable_model import ExecutableModelRegistry, replay_candidate
from tests.executable_model_fixtures import (
    CORRECT_SOURCE,
    WRONG_SOURCE,
    candidate,
    cases_from_shift_trace,
)


def test_a_deliberately_wrong_model_is_falsified_on_holdout(tmp_path: Path) -> None:
    cases = cases_from_shift_trace(tmp_path)
    assert any(c.split == "holdout" for c in cases)
    report = replay_candidate(candidate(WRONG_SOURCE), WRONG_SOURCE, cases)
    assert report.holdout_accuracy is not None and report.holdout_accuracy < 1.0
    assert report.counterexample_case_ids


def test_a_correct_model_matches_every_case(tmp_path: Path) -> None:
    cases = cases_from_shift_trace(tmp_path)
    report = replay_candidate(candidate(CORRECT_SOURCE), CORRECT_SOURCE, cases)
    assert report.train_accuracy == 1.0
    assert report.holdout_accuracy == 1.0
    assert report.counterexample_case_ids == ()


def test_a_revised_candidate_improves_holdout_accuracy_over_its_parent(
    tmp_path: Path,
) -> None:
    cases = cases_from_shift_trace(tmp_path)
    registry = ExecutableModelRegistry()
    v1 = candidate(WRONG_SOURCE, version=1)
    registry.add(v1)
    v2 = candidate(CORRECT_SOURCE, version=2, parent_version=1)
    registry.add(v2)
    report_v1 = replay_candidate(v1, WRONG_SOURCE, cases)
    report_v2 = replay_candidate(v2, CORRECT_SOURCE, cases)
    assert report_v1.holdout_accuracy is not None
    assert report_v2.holdout_accuracy is not None
    assert report_v2.holdout_accuracy > report_v1.holdout_accuracy
    assert registry.get("shift", 2).parent_version == 1
