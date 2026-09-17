"""Historical exact-transition checks for a declarative model."""

from typing import TYPE_CHECKING

from .contracts import ReplayCase, ReplayCheck

if TYPE_CHECKING:
    from .runtime import WorldModelRuntime


def replay_cases(
    runtime: "WorldModelRuntime", cases: tuple[ReplayCase, ...]
) -> tuple[ReplayCheck, ...]:
    checks = []
    for case in cases:
        status, predicted, _ = runtime.apply(case.before, case.action)
        checks.append(
            ReplayCheck(
                case_id=case.case_id,
                passed=status == "applied" and predicted == case.expected_after,
                predicted_hash=predicted.state_hash if predicted else None,
                expected_hash=case.expected_after.state_hash,
            )
        )
    return tuple(checks)
