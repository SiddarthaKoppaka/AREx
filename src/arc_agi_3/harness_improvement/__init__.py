"""External harness self-improvement loop (PR-10): failure packets,
candidate lineage, static checks, and a held-out promotion criterion.

This is measurement/bookkeeping scaffolding for a human-driven workflow
(propose a change in a separate branch, run checks, compare paired runs,
a human decides) - not an autonomous patch-generator. See
PR-10-report.md.
"""

from .contracts import (
    FailurePacket,
    HarnessChangeCandidate,
    PromotionDecision,
    StaticCheckResult,
)
from .failure_packet import build_failure_packet
from .promotion import decide_promotion
from .registry import HarnessChangeRegistry
from .static_checks import run_static_checks

__all__ = [
    "FailurePacket",
    "HarnessChangeCandidate",
    "HarnessChangeRegistry",
    "PromotionDecision",
    "StaticCheckResult",
    "build_failure_packet",
    "decide_promotion",
    "run_static_checks",
]
