"""Executable transition-model candidates: contract, static interface
check, and sandboxed execution. See ADR-V2-002/003/006 and
PR-03-scoping.md. Alongside, not replacing, `world_model/`."""

from .contracts import ExecutableModelCandidate, ExecutionProfile
from .execution import ExecutionResult
from .hashing import source_sha256
from .interface_check import InterfaceCheckError, check_interface
from .registry import ExecutableModelRegistry
from .replay import replay_candidate
from .replay_cases import build_transition_cases
from .replay_contracts import CaseResult, ModelTestReport, TransitionCase
from .sandbox_executor import run_candidate

__all__ = [
    "CaseResult",
    "ExecutableModelCandidate",
    "ExecutableModelRegistry",
    "ExecutionProfile",
    "ExecutionResult",
    "InterfaceCheckError",
    "ModelTestReport",
    "TransitionCase",
    "build_transition_cases",
    "check_interface",
    "replay_candidate",
    "run_candidate",
    "source_sha256",
]
