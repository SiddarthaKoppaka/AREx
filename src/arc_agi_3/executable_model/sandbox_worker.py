"""Subprocess entry point for one sandboxed candidate-function call.

Invoked as `python3 -m arc_agi_3.executable_model.sandbox_worker`. This
process IS the untrusted boundary: it sets OS resource limits before
touching candidate source, then execs that source in a restricted
namespace. Reads one JSON request from stdin, writes one JSON response to
stdout, and never raises - every failure becomes a typed response, since
an uncaught exception here would just be a less informative crash.

A CPU-limit breach (RLIMIT_CPU) kills this process via SIGXCPU before it
can respond at all - the parent (`sandbox_executor.py`) must detect that
by exit signal, not by waiting for a "timeout" status from here.
"""

import json
import resource
import sys
from contextlib import suppress

from .restricted_builtins import safe_builtins


def _set_limits(cpu_seconds: float, memory_bytes: int) -> None:
    cpu = max(1, int(cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    with suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NOFILE, (8, 8))
    with suppress(ValueError, OSError):
        # Not settable on every platform (e.g. macOS); parent polls RSS instead.
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def _run(request: dict[str, object]) -> dict[str, object]:
    source = str(request["source"])
    function = str(request["function"])
    args = request["args"]
    assert isinstance(args, list)
    cpu_seconds = request.get("cpu_seconds", 2.0)
    memory_bytes = request.get("memory_bytes", 256 * 1024 * 1024)
    assert isinstance(cpu_seconds, int | float) and isinstance(memory_bytes, int)
    _set_limits(float(cpu_seconds), memory_bytes)
    namespace: dict[str, object] = {"__builtins__": safe_builtins()}
    try:
        exec(compile(source, "<candidate>", "exec"), namespace)
        target = namespace.get(function)
        if not callable(target):
            return {"status": "interface_error", "message": f"{function!r} not defined"}
        result = target(*args)
    except Exception as error:
        return {
            "status": "runtime_error",
            "message": f"{type(error).__name__}: {error}",
        }
    serialized = json.dumps(result)
    limit = request.get("output_bytes", 65_536)
    assert isinstance(limit, int)
    if len(serialized.encode("utf-8")) > limit:
        return {"status": "output_too_large", "message": f"{len(serialized)} bytes"}
    return {"status": "ok", "output": result}


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        response = _run(request)
    except Exception as error:
        response = {
            "status": "runtime_error",
            "message": f"{type(error).__name__}: {error}",
        }
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
