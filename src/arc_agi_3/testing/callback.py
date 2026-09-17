"""Deterministic fixtures for externally-owned environment loops."""

from pathlib import Path
from types import SimpleNamespace

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.deployment.callback_environment import CallbackEnvironment
from arc_agi_3.deployment.kaggle_bridge import KaggleAgentBridge
from arc_agi_3.runtime import build_session

from .fakes import ScriptedModel


def callback_frame(
    position: int, state: EnvironmentState, actions: list[int]
) -> object:
    return SimpleNamespace(
        game_id="callback-game",
        frame=[[[position, 0]]],
        state=state,
        levels_completed=int(state is EnvironmentState.WIN),
        win_levels=1,
        available_actions=actions,
        guid=f"frame-{position}-{state}",
        full_reset=False,
    )


def callback_bridge(
    output: Path, run_id: str, decisions: list[CognitiveDecision]
) -> KaggleAgentBridge:
    config = RunConfig(
        run_id=run_id,
        experiment_id="callback-test",
        game_id="callback-game",
        output_dir=output,
        max_turns=4,
    )
    environment = CallbackEnvironment("callback-game", "test")
    session = build_session(config, environment, ScriptedModel(decisions))
    return KaggleAgentBridge("callback-game", session)
