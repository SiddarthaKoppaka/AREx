"""Shared fixtures for PR-09 skills tests."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, Resource
from arc_agi_3.contracts.execution import ActionChunk, ChunkStep
from arc_agi_3.contracts.observation import Action
from arc_agi_3.runtime import build_runner
from arc_agi_3.skills import Skill, SkillPrecondition
from arc_agi_3.testing import FakeLineEnvironment
from arc_agi_3.testing.epistemic import ReactiveModel

CHUNK = ActionChunk(
    chunk_id="walk-3",
    steps=tuple(ChunkStep(action=Action(action_id=1)) for _ in range(3)),
)


def skill(**overrides: object) -> Skill:
    fields: dict[str, object] = {
        "skill_id": "walk",
        "version": 1,
        "game_id": "fake-line",
        "chunk_template": CHUNK,
        "precondition": SkillPrecondition(),
        "evidence_refs": ("event-1",),
    }
    fields.update(overrides)
    return Skill(**fields)


def stop(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(assessment="x", intent="x", mode=DecisionMode.STOP)


def _model_calls(tmp_path: Path, run_id: str, model: ReactiveModel) -> int:
    config = RunConfig(run_id=run_id, experiment_id="test", output_dir=tmp_path)
    runner = build_runner(
        config, FakeLineEnvironment(target=10), model, raise_on_failure=True
    )
    runner.run()
    return runner.io.ledger.snapshot().consumed[Resource.MODEL_CALLS]


def run_single_actions(tmp_path: Path, count: int) -> int:
    def execute(context: AgentContext) -> CognitiveDecision:
        return CognitiveDecision(
            assessment="x",
            intent="x",
            mode=DecisionMode.EXECUTE,
            action=Action(action_id=1),
        )

    return _model_calls(tmp_path, "single", ReactiveModel([execute] * count + [stop]))


def run_skill_chunk(tmp_path: Path) -> int:
    def execute_chunk(context: AgentContext) -> CognitiveDecision:
        return CognitiveDecision(
            assessment="x", intent="x", mode=DecisionMode.EXECUTE, action_chunk=CHUNK
        )

    return _model_calls(tmp_path, "chunk", ReactiveModel([execute_chunk, stop]))
