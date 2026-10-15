"""Derived research indexes and exports; raw traces remain authoritative."""

from .aggregate import AggregateResult, aggregate_rows
from .baseline_report import build_report
from .export import export_rows, run_row
from .index import TraceIndex
from .report_comparison import compare_reports
from .report_markdown import render_markdown

__all__ = [
    "AggregateResult",
    "TraceIndex",
    "aggregate_rows",
    "build_report",
    "compare_reports",
    "export_rows",
    "render_markdown",
    "run_row",
]
