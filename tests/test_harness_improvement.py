"""PR-10 gate: one candidate is demonstrably accepted and one rejected,
with full reproducible evaluation (reusing PR-00's BaselineReport/
compare_reports), parent lineage, and no authorization/trace regression."""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.harness_improvement import (
    HarnessChangeCandidate,
    HarnessChangeRegistry,
    build_failure_packet,
    decide_promotion,
)
from arc_agi_3.research.report_comparison import compare_reports
from arc_agi_3.trace.store import JsonlEventStore
from tests.harness_improvement_fixtures import (
    control_report,
    improving_candidate_report,
    regressing_candidate_report,
)

HASH = "0" * 64


def test_a_regressing_candidate_is_rejected_on_verification_failures(
    tmp_path: Path,
) -> None:
    comparison = compare_reports(
        control_report(tmp_path), regressing_candidate_report(tmp_path)
    )
    decision = decide_promotion("c1", comparison, "verification_failures")
    assert decision.accepted is False
    assert (
        decision.candidate_value is not None
        and decision.candidate_value > decision.control_value
    )


def test_an_improving_candidate_is_accepted_on_model_calls(tmp_path: Path) -> None:
    comparison = compare_reports(
        control_report(tmp_path), improving_candidate_report(tmp_path)
    )
    decision = decide_promotion("c2", comparison, "model_calls")
    assert decision.accepted is True
    assert (
        decision.candidate_value is not None
        and decision.candidate_value < decision.control_value
    )


def test_failure_packet_is_derived_from_the_real_report_not_invented(
    tmp_path: Path,
) -> None:
    packet = build_failure_packet(regressing_candidate_report(tmp_path))
    assert "did not match" in packet.failure_summary
    assert packet.metrics_snapshot["verification_failures"] == 1


def test_registry_enforces_parent_lineage_and_records_the_decision() -> None:
    registry = HarnessChangeRegistry()
    v1 = HarnessChangeCandidate(
        candidate_id="c1",
        version=1,
        diff_sha256=HASH,
        scope_description="narrow prompt tweak",
        failure_packet_refs=("pkt-1",),
    )
    registry.add(v1)
    registry.record_decision("c1", 1, accepted=False)
    assert registry.get("c1", 1).promotion_status == "rejected"
    v2 = HarnessChangeCandidate(
        candidate_id="c1",
        version=2,
        parent_version=1,
        diff_sha256=HASH,
        scope_description="revised prompt tweak",
        failure_packet_refs=("pkt-1",),
    )
    registry.add(v2)
    registry.record_decision("c1", 2, accepted=True)
    assert registry.get("c1", 2).promotion_status == "accepted"


def test_no_authorization_or_trace_regression_from_building_a_report(
    tmp_path: Path,
) -> None:
    control_report(tmp_path)
    events = JsonlEventStore(
        tmp_path / "control" / "events.jsonl", "a", "e", "b"
    ).read()
    assert audit_agency_boundary(events) == ()
