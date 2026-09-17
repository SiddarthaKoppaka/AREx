"""Soft interruption occurs only for an LM-authorized condition."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.execution import ActionChunk, ChunkStep
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel


def test_lm_authored_change_threshold_soft_interrupts(tmp_path: Path) -> None:
    chunk = ActionChunk(
        chunk_id="soft-chunk",
        steps=(
            ChunkStep(action=Action(action_id=1)),
            ChunkStep(action=Action(action_id=1)),
        ),
        soft_interrupt_changed_cells=1,
    )
    execute = CognitiveDecision(
        assessment="authorize threshold",
        intent="interrupt after a large change",
        mode=DecisionMode.EXECUTE,
        action_chunk=chunk,
    )
    stop = CognitiveDecision(
        assessment="inspect", intent="stop", mode=DecisionMode.STOP
    )
    config = RunConfig(run_id="soft-chunk", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeLineEnvironment(), ScriptedModel([execute, stop]))
    result = runner.run()
    interrupts = [
        event
        for event in runner.events.read()
        if event.event_type is EventType.INTERRUPT
    ]
    assert result.metrics.environment_actions == 1
    assert interrupts[0].payload["status"] == "soft_interrupt"
