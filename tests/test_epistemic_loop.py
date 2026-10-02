"""End to end: hypothesize, predict, act, mismatch, evidence, explicit revision."""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.cognition import BeliefUpdate, Hypothesis
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import BeliefOperation, EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, ReactiveModel
from tests.epistemic_policies import hypothesize, probe, stop


def revise(context: AgentContext) -> CognitiveDecision:
    entry = context.hypothesis_ledger[0]
    contradiction = entry.unacknowledged_contradictions[0]
    assert context.latest_transition is not None
    moved = context.latest_transition.translations[0].value
    update = BeliefUpdate(
        hypothesis_id="h1",
        expected_version=entry.hypothesis.version,
        operation=BeliefOperation.CONTRADICT,
        strength="strong",
        evidence_refs=(contradiction,),
        confidence=0.2,
    )
    rival = Hypothesis(
        hypothesis_id="h2",
        version=1,
        claim=f"Value {moved} responds to action 1.",
        probability=0.6,
        evidence_refs=(contradiction,),
        supersedes=("h1",),
    )
    return probe("h2", moved).model_copy(
        update={"hypothesis_updates": (update,), "hypothesis_proposals": (rival,)}
    )


def test_mismatch_becomes_evidence_and_model_revises(tmp_path: Path) -> None:
    model = ReactiveModel([hypothesize, revise, stop])
    config = RunConfig(run_id="loop", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    result = runner.run()
    after_mismatch, after_revision = model.contexts[1], model.contexts[2]
    assert after_mismatch.latest_verification is not None
    assert after_mismatch.latest_verification.status == "mismatched"
    assert after_mismatch.latest_verification.mismatches == ("translation",)
    assert after_mismatch.hypotheses[0].probability == 0.7
    assert after_mismatch.unresolved_contradictions
    h1 = next(h for h in after_revision.hypotheses if h.hypothesis_id == "h1")
    assert h1.probability == 0.2 and h1.superseded_by == "h2"
    cited = after_mismatch.hypothesis_ledger[0].unacknowledged_contradictions
    assert h1.contradicting_refs == cited
    assert after_mismatch.hypothesis_ledger[0].tests[0].verification_event_id in cited
    h2 = next(h for h in after_revision.hypotheses if h.hypothesis_id == "h2")
    assert h2.created_turn == 2 and h2.supersedes == ("h1",)
    assert after_revision.unresolved_contradictions == ()
    assert after_revision.latest_verification is not None
    assert after_revision.latest_verification.status == "matched"
    events = runner.events.read()
    assert audit_agency_boundary(events) == ()
    assert sum(e.event_type is EventType.EXPERIMENT for e in events) == 2
    metrics = result.metrics
    assert (metrics.prediction_mismatches, metrics.predictions_matched) == (1, 1)
    assert (metrics.hypothesis_revisions, metrics.experiments) == (1, 2)
    assert (metrics.environment_actions, metrics.model_calls) == (2, 3)
    assert metrics.verification_failures == 1 and metrics.unjustified_repeats == 0
