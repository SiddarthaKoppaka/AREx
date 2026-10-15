"""PR-09 gate: a validated skill reduces a repeated sequence's model-call
cost, and its instantiated chunk can never bypass the action firewall -
it only ever produces a normal ActionChunk, which goes through the
existing, unmodified per-step authorization path and still hard-stops on
a violated precondition exactly like any other chunk."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType, ExecutionStatus
from arc_agi_3.contracts.execution import ActionChunk, ChunkStep
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment
from arc_agi_3.testing.epistemic import ReactiveModel
from tests.skills_fixtures import run_single_actions, run_skill_chunk, stop


def test_a_validated_skill_reduces_model_call_cost_for_a_repeated_sequence(
    tmp_path: Path,
) -> None:
    separate_calls = run_single_actions(tmp_path / "separate", 3)
    chunked_calls = run_skill_chunk(tmp_path / "chunked")
    assert chunked_calls < separate_calls
    assert separate_calls - chunked_calls == 2  # one decision authorizes all 3 steps


def test_a_skill_chunk_still_hard_stops_on_a_violated_per_step_precondition(
    tmp_path: Path,
) -> None:
    stale_chunk = ActionChunk(
        chunk_id="stale",
        steps=(
            ChunkStep(action=Action(action_id=1)),
            ChunkStep(
                action=Action(action_id=1), required_before_hash="not-a-real-hash"
            ),
        ),
    )

    def execute_chunk(context: AgentContext) -> CognitiveDecision:
        return CognitiveDecision(
            assessment="x",
            intent="x",
            mode=DecisionMode.EXECUTE,
            action_chunk=stale_chunk,
        )

    config = RunConfig(run_id="stale", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config,
        FakeLineEnvironment(target=10),
        ReactiveModel([execute_chunk, stop]),
        raise_on_failure=True,
    )
    runner.run()
    events = runner.events.read()
    interrupts = [e for e in events if e.event_type is EventType.INTERRUPT]
    assert any(e.payload.get("status") == ExecutionStatus.HARD_STOP for e in interrupts)
    actions = [e for e in events if e.event_type is EventType.ACTION]
    assert len(actions) == 1  # the firewall stopped the second, stale-precondition step
