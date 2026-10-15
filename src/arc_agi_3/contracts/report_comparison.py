"""Control-vs-candidate deltas over two `BaselineReport`s."""

from .base import Contract
from .baseline_report import BaselineReport


class MetricDelta(Contract):
    """`delta` is `None`, never `0`, whenever either side is `None` — a
    missing value must never be silently coerced into a real zero."""

    name: str
    control: float | int | None
    candidate: float | int | None
    delta: float | int | None


class ReportComparison(Contract):
    control: BaselineReport
    candidate: BaselineReport
    deltas: tuple[MetricDelta, ...]
