"""PR-06: the full scientist-loop causal chain, end to end, with a fixture
LM. SYNTHETIC: a scripted model, not a real LM - demonstrates the causal
machinery works, not that a real Qwen run learns this way. See
test_scientist_loop_falsification.py for the offline replay half.

Two competing hypotheses -> a discriminating authorized action -> the
verifier records the actual outcome -> a later decision reuses the
verified fact instead of repeating the real experiment. Composes PR-02's
audit and the existing runner/evidence machinery; no new infrastructure.
"""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, ReactiveModel
from tests.scientist_loop_fixtures import (
    propose_and_test,
    record_fact,
    use_prior_knowledge,
)


def test_full_scientist_loop_causal_chain(tmp_path: Path) -> None:
    model = ReactiveModel([propose_and_test, record_fact, use_prior_knowledge])
    config = RunConfig(run_id="scientist", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    runner.run()
    events = runner.events.read()

    assert audit_agency_boundary(events) == ()

    by_type = {t: [e for e in events if e.event_type is t] for t in EventType}
    hypothesis_event = by_type[EventType.HYPOTHESIS][0]
    decision_event = by_type[EventType.MODEL_DECISION][0]
    action_event = by_type[EventType.ACTION][0]
    experiment_event = by_type[EventType.EXPERIMENT][0]
    transition_event = by_type[EventType.TRANSITION][0]
    verification_event = by_type[EventType.VERIFICATION][0]
    assert hypothesis_event.causal_refs == (decision_event.event_id,)
    assert action_event.causal_refs[0] == decision_event.event_id
    assert experiment_event.causal_refs[:2] == (
        decision_event.event_id,
        action_event.event_id,
    )
    assert verification_event.causal_refs == (
        decision_event.event_id,
        transition_event.event_id,
    )
