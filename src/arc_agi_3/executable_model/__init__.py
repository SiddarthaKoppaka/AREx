"""Executable transition-model candidates: contract, static interface
check, and sandboxed execution. See ADR-V2-002/003/006 and
PR-03-scoping.md. Alongside, not replacing, `world_model/`."""

from .contracts import ExecutableModelCandidate, ExecutionProfile
from .execution import ExecutionResult
from .hashing import source_sha256
from .interface_check import InterfaceCheckError, check_interface
from .sandbox_executor import run_candidate

__all__ = [
    "ExecutableModelCandidate",
    "ExecutionProfile",
    "ExecutionResult",
    "InterfaceCheckError",
    "check_interface",
    "run_candidate",
    "source_sha256",
]
