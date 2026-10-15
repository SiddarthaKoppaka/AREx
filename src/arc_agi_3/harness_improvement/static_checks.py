"""Run the project's own static checks against a candidate's working
tree. This runs ordinary dev tooling on code a human is about to review
- not the untrusted-code sandbox PR-03 built; no special isolation needed.
"""

import subprocess
from pathlib import Path

from .contracts import StaticCheckResult

_CHECKS = (
    ("ruff-check", ("uv", "run", "ruff", "check", ".")),
    ("ruff-format", ("uv", "run", "ruff", "format", "--check", ".")),
    ("mypy", ("uv", "run", "mypy")),
)


def _run_one(name: str, command: tuple[str, ...], repo_root: Path) -> StaticCheckResult:
    try:
        result = subprocess.run(
            command, cwd=repo_root, capture_output=True, text=True, timeout=600
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        return StaticCheckResult(
            check=name, passed=False, output_tail=str(error)[:4000]
        )
    tail = (result.stdout + result.stderr)[-4000:]
    return StaticCheckResult(
        check=name, passed=result.returncode == 0, output_tail=tail
    )


def run_static_checks(repo_root: Path) -> tuple[StaticCheckResult, ...]:
    return tuple(_run_one(name, command, repo_root) for name, command in _CHECKS)
