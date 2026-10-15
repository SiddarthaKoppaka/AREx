"""The PR-03 gate: a correct program runs, and every adversarial program
is contained - infinite loop, huge output, filesystem/network/import
attempts, global mutation, and crashes all end up as a typed result, not
a hang, a crash of this test process, or unbounded resource use."""

from arc_agi_3.executable_model.contracts import (
    ExecutableModelCandidate,
    ExecutionProfile,
)
from arc_agi_3.executable_model.sandbox_executor import run_candidate

FAST_PROFILE = ExecutionProfile(
    cpu_seconds=1, wall_seconds=2, memory_bytes=64 * 1024 * 1024
)


def candidate(source: str) -> ExecutableModelCandidate:
    return ExecutableModelCandidate(
        model_id="m",
        version=1,
        source_artifact_ref="artifacts/x.json",
        source_sha256="0" * 64,
        execution_profile=FAST_PROFILE,
    )


def test_a_correct_deterministic_program_runs() -> None:
    source = (
        "def encode_observation(o):\n    return o\n"
        "def transition(state, action):\n    return {'x': state['x'] + action['d']}\n"
        "def terminal_or_goal(state):\n    return state['x'] >= 10\n"
    )
    result = run_candidate(candidate(source), source, "transition", {"x": 1}, {"d": 2})
    assert result.status == "ok"
    assert result.output == {"x": 3}


def test_an_infinite_busy_loop_is_killed_not_hung() -> None:
    source = (
        "def transition(state, action):\n    x = 0\n    while True:\n        x += 1\n"
    )
    result = run_candidate(candidate(source), source, "transition", {}, {})
    assert result.status == "timeout"


def test_a_wall_clock_sleep_loop_is_also_killed() -> None:
    source = (
        "def transition(state, action):\n"
        "    import time\n"
        "    time.sleep(999)\n"
        "    return {}\n"
    )
    result = run_candidate(candidate(source), source, "transition", {}, {})
    assert result.status in ("timeout", "disallowed_import", "runtime_error")


def test_huge_output_is_rejected_not_streamed_back() -> None:
    source = "def transition(state, action):\n    return {'big': 'x' * 10_000_000}\n"
    result = run_candidate(candidate(source), source, "transition", {}, {})
    assert result.status == "output_too_large"


def test_filesystem_write_attempt_is_blocked() -> None:
    source = (
        "def transition(state, action):\n"
        "    with open('/tmp/sandbox_escape_attempt.txt', 'w') as f:\n"
        "        f.write('x' * 1_000_000)\n"
        "    return {}\n"
    )
    result = run_candidate(candidate(source), source, "transition", {}, {})
    assert result.status == "runtime_error"


def test_network_import_attempt_is_blocked() -> None:
    source = (
        "def transition(state, action):\n"
        "    import socket\n"
        "    socket.socket()\n"
        "    return {}\n"
    )
    result = run_candidate(candidate(source), source, "transition", {}, {})
    assert result.status == "runtime_error"
    assert "import" in (result.message or "").lower()


def test_global_mutation_attempt_is_contained_to_the_subprocess() -> None:
    source = (
        "shared = {'count': 0}\n"
        "def transition(state, action):\n"
        "    global shared\n"
        "    shared['count'] += 1\n"
        "    return shared\n"
    )
    first = run_candidate(candidate(source), source, "transition", {}, {})
    second = run_candidate(candidate(source), source, "transition", {}, {})
    assert first.status == "ok" and second.status == "ok"
    assert first.output == second.output == {"count": 1}


def test_a_crashing_program_returns_a_typed_runtime_error() -> None:
    source = "def transition(state, action):\n    return 1 / 0\n"
    result = run_candidate(candidate(source), source, "transition", {}, {})
    assert result.status == "runtime_error"
    assert "ZeroDivisionError" in (result.message or "")
