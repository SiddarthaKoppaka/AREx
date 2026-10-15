"""Typed outcome of one sandboxed candidate-function call.

Every failure mode is a typed status, never an uncaught exception: a
candidate that times out, allocates too much memory, writes too much
output, imports a disallowed module, or crashes must report that fact to
the harness exactly like a successful call does, not propagate upward.
"""

from typing import Literal

from pydantic import JsonValue

from arc_agi_3.contracts.base import Contract

ExecutionStatus = Literal[
    "ok",
    "timeout",
    "memory_exceeded",
    "output_too_large",
    "disallowed_import",
    "runtime_error",
    "syntax_error",
    "interface_error",
]


class ExecutionResult(Contract):
    status: ExecutionStatus
    output: JsonValue = None
    message: str | None = None
    cpu_seconds: float | None = None
    wall_seconds: float | None = None
