"""PR-06: offline replay falsifies the wrong hypothesis's candidate and
confirms the correct one against the same recorded trace, then a
baseline report is generated against it. SYNTHETIC: fixture LM, not a
real Qwen run - see PR-06-report.md."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.evidence.history import transition_records
from arc_agi_3.executable_model import build_transition_cases, replay_candidate
from arc_agi_3.research.baseline_report import build_report
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, ReactiveModel
from tests.executable_model_fixtures import CORRECT_SOURCE, candidate
from tests.scientist_loop_fixtures import (
    WRONG_SHIFT_SOURCE,
    propose_and_test,
    record_fact,
    use_prior_knowledge,
)


def test_offline_replay_falsifies_the_wrong_candidate_and_confirms_the_right_one(
    tmp_path: Path,
) -> None:
    model = ReactiveModel([propose_and_test, record_fact, use_prior_knowledge])
    config = RunConfig(run_id="scientist", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    runner.run()
    events = runner.events.read()

    cases = build_transition_cases(
        transition_records(events), seed=0, holdout_fraction=0.0
    )
    correct_report = replay_candidate(candidate(CORRECT_SOURCE), CORRECT_SOURCE, cases)
    wrong_report = replay_candidate(
        candidate(WRONG_SHIFT_SOURCE, version=2, parent_version=1),
        WRONG_SHIFT_SOURCE,
        cases,
    )
    assert correct_report.train_accuracy == 1.0
    assert wrong_report.train_accuracy == 0.0
    assert wrong_report.counterexample_case_ids

    report = build_report(tmp_path / "scientist")
    assert (
        report.stop_reason == "agent_stop"
    )  # SYNTHETIC: fixture LM, not a real Qwen run
