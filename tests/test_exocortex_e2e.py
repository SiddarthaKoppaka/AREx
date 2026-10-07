"""The core agent requests consultations on demand; a Critic disagreement
is still visible before the core agent acts, and the audit confirms the
executed action is the only thing with authority. See ADR-0009."""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment
from tests.exocortex_e2e_fixtures import three_turn_core_agent
from tests.exocortex_e2e_outputs import specialist_outputs
from tests.generation_fixtures import ScriptedBackend


def test_core_agent_consults_on_demand_and_sees_the_disagreement(
    tmp_path: Path,
) -> None:
    specialists = ScriptedBackend(specialist_outputs())
    core_agent = three_turn_core_agent()
    config = RunConfig(
        run_id="exocortex-e2e", experiment_id="test", output_dir=tmp_path
    )
    runner = build_runner(
        config,
        FakeLineEnvironment(target=1),
        core_agent,
        specialist_backend=specialists,
        raise_on_failure=True,
    )
    result = runner.run()

    assert result.stop_reason == "terminal"
    assert result.metrics.succeeded is True
    assert result.metrics.specialist_reports_generated == 2
    assert result.metrics.specialist_reviews_generated == 1
    assert result.metrics.flagged_contradictions_raised == 1

    # The core agent's 3rd/4th calls already see both the hypothesis and the
    # critic's disagreement, with zero re-consultation.
    synthesis = core_agent.contexts[-1].cognitive_synthesis
    assert synthesis is not None
    assert synthesis.candidate_hypotheses[0].claim == "value 11 is a goal"
    assert synthesis.flagged_contradictions == (
        "S-critic on S-hypothesis: evidence only proves cells changed, not goal status",
    )

    events = runner.events.read()
    kinds = [e.event_type for e in events]
    assert kinds.count(EventType.SPECIALIST_REPORT) == 2
    assert kinds.count(EventType.SPECIALIST_REVIEW) == 1
    assert audit_agency_boundary(events) == ()
