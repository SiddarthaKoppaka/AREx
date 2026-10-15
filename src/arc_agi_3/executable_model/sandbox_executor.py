"""Parent-side orchestrator for one sandboxed candidate-function call.

Spawns `sandbox_worker.py` as a fresh subprocess per call - no state
persists between calls, and a crashed/killed worker can never affect the
harness process. Three independent safety nets beyond the worker's own
resource limits: a wall-clock watchdog (catches sleeps/IO waits
RLIMIT_CPU can't see), a bounded read of stdout, and the RSS-polling
memory backstop (`memory_watchdog.py`) where RLIMIT_AS cannot be set.
"""

import json
import subprocess
import sys
import threading
import time

from .contracts import ExecutableModelCandidate
from .execution import ExecutionResult
from .memory_watchdog import watch


def run_candidate(
    candidate: ExecutableModelCandidate, source: str, function: str, *args: object
) -> ExecutionResult:
    profile = candidate.execution_profile
    request = json.dumps(
        {
            "source": source,
            "function": function,
            "args": list(args),
            "cpu_seconds": profile.cpu_seconds,
            "memory_bytes": profile.memory_bytes,
            "output_bytes": profile.output_bytes,
        }
    )
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-m", "arc_agi_3.executable_model.sandbox_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    exceeded_memory = threading.Event()
    stop_watchdog = threading.Event()
    watchdog = threading.Thread(
        target=watch,
        args=(process.pid, profile.memory_bytes, exceeded_memory, stop_watchdog),
        daemon=True,
    )
    watchdog.start()
    try:
        stdout, _ = process.communicate(input=request, timeout=profile.wall_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        stop_watchdog.set()
        return ExecutionResult(
            status="timeout", wall_seconds=time.monotonic() - started
        )
    stop_watchdog.set()
    elapsed = time.monotonic() - started
    if exceeded_memory.is_set():
        return ExecutionResult(status="memory_exceeded", wall_seconds=elapsed)
    if process.returncode is not None and process.returncode < 0:
        return ExecutionResult(
            status="timeout", wall_seconds=elapsed, message="killed by signal"
        )
    return _parse(stdout, elapsed)


def _parse(stdout: str | None, elapsed: float) -> ExecutionResult:
    try:
        payload = json.loads(stdout or "")
    except json.JSONDecodeError:
        return ExecutionResult(
            status="runtime_error",
            message="worker produced no valid response",
            wall_seconds=elapsed,
        )
    return ExecutionResult(
        status=payload.get("status", "runtime_error"),
        output=payload.get("output"),
        message=payload.get("message"),
        wall_seconds=elapsed,
    )
