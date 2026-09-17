"""Chunks execute only the LM-authorized prefix under objective guards."""

from pathlib import Path

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import CognitiveDecision, ExpectedOutcome
from arc_agi_3.contracts.enums import DecisionMode, EnvironmentState, EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.execution import ActionChunk, ChunkStep
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.runtime.divergence import find_first_divergence
from arc_agi_3.runtime.state import RunResult
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def decision(chunk: ActionChunk) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="The LM authorizes this exact chunk.",
        intent="Execute while its contract remains valid.",
        mode=DecisionMode.EXECUTE,
        action_chunk=chunk,
    )


def events_for(
    tmp_path: Path, first: CognitiveDecision
) -> tuple[RunResult, list[EventEnvelope]]:
    stop = CognitiveDecision(
        assessment="inspect", intent="stop", mode=DecisionMode.STOP
    )
    chunk = first.action_chunk
    assert chunk is not None
    config = RunConfig(
        run_id=chunk.chunk_id, experiment_id="chunks", output_dir=tmp_path
    )
    runner = build_runner(config, FakeLineEnvironment(), ScriptedModel([first, stop]))
    result = runner.run()
    return result, runner.events.read()


def test_chunk_reaches_terminal_without_an_extra_model_call(tmp_path: Path) -> None:
    chunk = ActionChunk(
        chunk_id="terminal-chunk",
        steps=(
            ChunkStep(action=Action(action_id=1)),
            ChunkStep(action=Action(action_id=1)),
        ),
    )
    result, events = events_for(tmp_path, decision(chunk))
    assert result.stop_reason == "terminal"
    assert result.metrics.environment_actions == 2
    assert result.metrics.model_calls == 1
    assert audit_agency_boundary(events) == ()


def test_prediction_mismatch_hard_stops_remaining_prefix(tmp_path: Path) -> None:
    chunk = ActionChunk(
        chunk_id="mismatch-chunk",
        steps=(
            ChunkStep(
                action=Action(action_id=1),
                expected_outcome=ExpectedOutcome(state=EnvironmentState.WIN),
            ),
            ChunkStep(action=Action(action_id=1)),
        ),
    )
    result, events = events_for(tmp_path, decision(chunk))
    interrupts = [event for event in events if event.event_type is EventType.INTERRUPT]
    assert result.metrics.environment_actions == 1
    assert interrupts[0].payload["reason"] == "prediction_mismatch"
    divergence = find_first_divergence(events)
    assert divergence is not None and divergence.reason.startswith(
        "prediction_mismatch"
    )
    assert audit_agency_boundary(events) == ()


def test_stale_precondition_executes_no_action(tmp_path: Path) -> None:
    chunk = ActionChunk(
        chunk_id="stale-chunk",
        steps=(ChunkStep(action=Action(action_id=1), required_before_hash="stale"),),
    )
    result, events = events_for(tmp_path, decision(chunk))
    assert result.metrics.environment_actions == 0
    assert EventType.ACTION not in {event.event_type for event in events}
    assert EventType.INTERRUPT in {event.event_type for event in events}
