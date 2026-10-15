"""Test an LM-authored candidate against real recorded transitions.

The harness computes every result here; the LM only ever reads the
report. No predicted state is ever written back as if it were a real
observation - this module only reads existing cases and returns an
in-memory report, it never appends to the trace.
"""

from .contracts import ExecutableModelCandidate
from .replay_contracts import CaseResult, ModelTestReport, TransitionCase
from .sandbox_executor import run_candidate


def _run_case(
    candidate: ExecutableModelCandidate, source: str, case: TransitionCase
) -> CaseResult:
    before = run_candidate(
        candidate, source, "encode_observation", case.before.model_dump(mode="json")
    )
    if before.status != "ok":
        return CaseResult(
            case_id=case.case_id,
            split=case.split,
            status="error",
            error_category=before.status,
        )
    after = run_candidate(
        candidate, source, "encode_observation", case.after.model_dump(mode="json")
    )
    if after.status != "ok":
        return CaseResult(
            case_id=case.case_id,
            split=case.split,
            status="error",
            error_category=after.status,
        )
    predicted = run_candidate(
        candidate,
        source,
        "transition",
        before.output,
        case.action.model_dump(mode="json"),
    )
    if predicted.status != "ok":
        return CaseResult(
            case_id=case.case_id,
            split=case.split,
            status="error",
            error_category=predicted.status,
        )
    if predicted.output == after.output:
        return CaseResult(case_id=case.case_id, split=case.split, status="match")
    return CaseResult(case_id=case.case_id, split=case.split, status="mismatch")


def replay_candidate(
    candidate: ExecutableModelCandidate, source: str, cases: tuple[TransitionCase, ...]
) -> ModelTestReport:
    results = [_run_case(candidate, source, case) for case in cases]
    return _aggregate(candidate, results)


def _aggregate(
    candidate: ExecutableModelCandidate, results: list[CaseResult]
) -> ModelTestReport:
    train = [r for r in results if r.split == "train"]
    holdout = [r for r in results if r.split == "holdout"]
    train_mismatches = sum(r.status == "mismatch" for r in train)
    holdout_mismatches = sum(r.status == "mismatch" for r in holdout)
    errors = sum(r.status == "error" for r in results)
    counterexamples = tuple(r.case_id for r in results if r.status == "mismatch")
    return ModelTestReport(
        model_id=candidate.model_id,
        model_version=candidate.version,
        total_cases=len(results),
        train_cases=len(train),
        holdout_cases=len(holdout),
        train_mismatches=train_mismatches,
        holdout_mismatches=holdout_mismatches,
        errors=errors,
        train_accuracy=_accuracy(train, train_mismatches),
        holdout_accuracy=_accuracy(holdout, holdout_mismatches),
        counterexample_case_ids=counterexamples,
    )


def _accuracy(split_results: list[CaseResult], mismatches: int) -> float | None:
    scored = len(split_results) - sum(r.status == "error" for r in split_results)
    return (scored - mismatches) / scored if scored else None
