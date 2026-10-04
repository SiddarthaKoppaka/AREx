"""Hypotheses are model-owned; the harness only stores and links evidence."""

from pathlib import Path

import pytest

from arc_agi_3.cognition import BeliefStore
from arc_agi_3.config import BeliefConfig, RunConfig
from arc_agi_3.contracts.cognition import BeliefUpdate, Hypothesis
from arc_agi_3.contracts.enums import BeliefOperation, EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, ReactiveModel
from tests.epistemic_policies import hypothesize, stop


def claim(key: str, **extra: object) -> Hypothesis:
    return Hypothesis.model_validate(
        {"hypothesis_id": key, "version": 1, "claim": key, "probability": 0.5, **extra}
    )


def update(operation: BeliefOperation, **extra: object) -> BeliefUpdate:
    values = {
        "hypothesis_id": "h1",
        "expected_version": 1,
        "operation": operation,
        "strength": "strong",
        "evidence_refs": ("event-9",),
        **extra,
    }
    return BeliefUpdate.model_validate(values)


def test_model_assigned_confidence_and_evidence_provenance() -> None:
    store = BeliefStore(BeliefConfig())
    store.add_many((claim("h1"),), turn=3)
    assert store.current[0].created_turn == 3
    (revised,) = store.apply(update(BeliefOperation.CONTRADICT, confidence=0.35))
    assert revised.probability == 0.35 and revised.version == 2
    assert revised.contradicting_refs == ("event-9",) and revised.evidence_refs == ()
    supported = store.apply(
        update(BeliefOperation.ACCEPT, expected_version=2, prediction_ids=("p1",))
    )[0]
    assert supported.status == "accepted" and supported.probability == 0.35
    assert supported.evidence_refs == ("event-9",) and supported.prediction_ids == (
        "p1",
    )


def test_superseding_hypothesis_links_both_records() -> None:
    store = BeliefStore(BeliefConfig())
    store.add_many((claim("h1"),))
    store.add_many((claim("h2", supersedes=("h1",)),))
    current = {item.hypothesis_id: item for item in store.current}
    assert current["h1"].superseded_by == "h2" and current["h1"].status == "active"
    assert current["h1"].probability == 0.5
    with pytest.raises(ValueError, match="superseded"):
        store.add_many((claim("h3", supersedes=("missing",)),))


def test_failed_prediction_never_changes_model_confidence(tmp_path: Path) -> None:
    model = ReactiveModel([hypothesize, stop])
    config = RunConfig(run_id="own", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    assert runner.run().metrics.prediction_mismatches == 1
    (h1,) = runner.io.workspace.beliefs.current
    assert (h1.probability, h1.version, h1.contradicting_refs) == (0.7, 1, ())
    events = runner.events.read()
    assert not any(e.event_type is EventType.BELIEF_UPDATE for e in events)
    entry = model.contexts[1].hypothesis_ledger[0]
    assert [test.prediction_id for test in entry.tests] == ["p-h1"]
    assert entry.tests[0].status == "mismatched"
    assert entry.unacknowledged_contradictions == (
        entry.tests[0].verification_event_id,
    )
