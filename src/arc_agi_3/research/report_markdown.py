"""Human-readable rendering of a `ReportComparison`."""

from arc_agi_3.contracts.baseline_report import BaselineReport
from arc_agi_3.contracts.report_comparison import ReportComparison

_NOT_OBSERVED = "not observed"


def _cell(value: float | int | None) -> str:
    return _NOT_OBSERVED if value is None else str(value)


def _provenance(label: str, report: BaselineReport) -> str:
    return (
        f"- **{label}**: run_id={report.run_id} model={report.model} "
        f"seed={report.seed} config_hash={report.config_hash} "
        f"stop_reason={report.stop_reason or _NOT_OBSERVED}"
    )


def render_markdown(comparison: ReportComparison) -> str:
    lines = [
        "# Baseline comparison",
        "",
        _provenance("control", comparison.control),
        _provenance("candidate", comparison.candidate),
        "",
        "| metric | control | candidate | delta |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {delta.name} | {_cell(delta.control)} | {_cell(delta.candidate)} "
        f"| {_cell(delta.delta)} |"
        for delta in comparison.deltas
    )
    return "\n".join(lines) + "\n"
