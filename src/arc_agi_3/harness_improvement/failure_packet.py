"""Derive a FailurePacket from an existing BaselineReport - never invent
a failure signal that the trace-derived metrics don't actually show."""

from arc_agi_3.contracts.baseline_report import BaselineReport

from .contracts import FailurePacket

_SIGNALS = (
    ("context_ceiling_failures", "the prompt exceeded its hard input-token limit"),
    (
        "metrics.unjustified_tool_repeats",
        "the agent repeated tool calls with no new evidence",
    ),
    (
        "metrics.unjustified_repeats",
        "the agent repeated an experiment with no new evidence",
    ),
    (
        "metrics.verification_failures",
        "a prediction did not match the observed outcome",
    ),
)


def _most_salient_signal(report: BaselineReport) -> tuple[str, int]:
    for path, description in _SIGNALS:
        value = (
            getattr(report.metrics, path.removeprefix("metrics."))
            if path.startswith("metrics.")
            else getattr(report, path)
        )
        if value:
            return description, int(value)
    return "no known failure signal was present in this run's metrics", 0


def build_failure_packet(report: BaselineReport) -> FailurePacket:
    description, count = _most_salient_signal(report)
    return FailurePacket(
        run_id=report.run_id,
        game_id=report.game_id,
        stop_reason=report.stop_reason,
        failure_summary=f"{description} ({count} occurrence(s))",
        evidence_refs=(report.last_event_hash,),
        metrics_snapshot=report.metrics.model_dump(mode="json"),
    )
