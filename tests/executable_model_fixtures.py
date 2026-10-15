"""Shared fixtures for executable-model replay tests."""

from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.observation import Action
from arc_agi_3.evidence.history import transition_records
from arc_agi_3.executable_model import ExecutableModelCandidate, source_sha256
from arc_agi_3.executable_model.replay_cases import build_transition_cases
from arc_agi_3.executable_model.replay_contracts import TransitionCase
from arc_agi_3.runtime import build_runner
from arc_agi_3.testing.epistemic import FakeShiftEnvironment, ReactiveModel

CORRECT_SOURCE = (
    "def encode_observation(o):\n"
    "    row = o['frame'][0][2]\n"
    "    column = 0\n"
    "    for i in range(len(row)):\n"
    "        if row[i] == 3:\n"
    "            column = i\n"
    "            break\n"
    "    return {'column': column}\n"
    "def transition(state, action):\n"
    "    if action['action_id'] == 1:\n"
    "        return {'column': min(6, state['column'] + 1)}\n"
    "    return {'column': state['column']}\n"
    "def terminal_or_goal(state):\n"
    "    return False\n"
)

WRONG_SOURCE = CORRECT_SOURCE.replace(
    "def transition(state, action):\n"
    "    if action['action_id'] == 1:\n"
    "        return {'column': min(6, state['column'] + 1)}\n"
    "    return {'column': state['column']}\n",
    "def transition(state, action):\n    return {'column': state['column'] + 1}\n",
)


def advance(action_id: int) -> CognitiveDecision:
    return CognitiveDecision(
        assessment="advance",
        intent="advance",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=action_id),
    )


def stop(context: AgentContext) -> CognitiveDecision:
    return CognitiveDecision(assessment="done", intent="stop", mode=DecisionMode.STOP)


def record_shift_trace(tmp_path: Path, turns: int = 16) -> list:
    policies = [
        (lambda context, action_id=1 + index % 2: advance(action_id))
        for index in range(turns)
    ]
    model = ReactiveModel([*policies, stop])
    config = RunConfig(run_id="replay", experiment_id="test", output_dir=tmp_path)
    runner = build_runner(config, FakeShiftEnvironment(), model, raise_on_failure=True)
    runner.run()
    return runner.events.read()


def cases_from_shift_trace(
    tmp_path: Path, *, turns: int = 16, holdout_fraction: float = 0.3
) -> tuple[TransitionCase, ...]:
    events = record_shift_trace(tmp_path, turns)
    return build_transition_cases(
        transition_records(events), seed=0, holdout_fraction=holdout_fraction
    )


def candidate(source: str, **overrides: object) -> ExecutableModelCandidate:
    fields: dict[str, object] = {
        "model_id": "shift",
        "version": 1,
        "source_artifact_ref": "artifacts/x.json",
        "source_sha256": source_sha256(source),
    }
    fields.update(overrides)
    return ExecutableModelCandidate(**fields)
