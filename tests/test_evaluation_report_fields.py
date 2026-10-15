"""Pure trace-derivation functions for stop_reason/context_ceiling_failures."""

from pathlib import Path

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.evaluation_report_fields import context_ceiling_failures, stop_reason
from arc_agi_3.trace.store import JsonlEventStore


def test_stop_reason_is_none_when_run_finished_is_absent(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl", "r", "e", "b")
    store.append(EventType.OBSERVATION, "env", 0, {})
    assert stop_reason(store.read()) is None


def test_context_ceiling_failures_counts_only_input_token_limit_errors(
    tmp_path: Path,
) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl", "r", "e", "b")
    store.append(
        EventType.FAILURE, "runtime", 0, {"error_type": "InputTokenLimitError"}
    )
    store.append(EventType.FAILURE, "runtime", 0, {"error_type": "RuntimeError"})
    store.append(EventType.RUN_FINISHED, "runtime", 0, {"reason": "failure"})
    assert context_ceiling_failures(store.read()) == 1
