"""Same model, same games: Duck vs the full harness vs the action-only baseline.

    python scripts/duck_vs_harness.py <duck|full|action_only> <steps> <game>...

Prints one JSON line per game and saves it under runs/compare/.
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import arc_agi

from arc_agi_3.adapters import OpenRouterBackend, OpenRouterConfig
from arc_agi_3.adapters.action_only_model import ActionOnlyModelAdapter
from arc_agi_3.adapters.arc import ArcEnvironmentAdapter
from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.config import BudgetConfig, RunConfig
from arc_agi_3.contracts.enums import EnvironmentState, Resource
from arc_agi_3.duck.factory import make_agent
from arc_agi_3.duck.settings import DuckSettings
from arc_agi_3.runtime.factory import build_runner

MODEL = os.environ.get("DUCK_MODEL_NAME", "google/gemma-4-31b-it")
ROOT = Path("arc-prize-2026-arc-agi-3/environment_files")
OUT = Path(os.environ.get("DUCK_OUT", "runs/compare"))


def environment(game: str) -> ArcEnvironmentAdapter:
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ROOT),
        recordings_dir=tempfile.mkdtemp(),
    )
    return ArcEnvironmentAdapter(arcade.make(game, seed=0, save_recording=False))


def play_duck(game: str, steps: int) -> dict[str, object]:
    settings = DuckSettings(
        backend="openrouter",
        model_name=MODEL,
        use_images=True,
        reasoning_effort=os.environ.get("DUCK_REASONING") or None,  # type: ignore[arg-type]
        max_tokens=int(os.environ.get("DUCK_MAX_TOKENS", "2048")),
        trace_path=f"{OUT}/trace-{game}.jsonl",
    )
    agent = make_agent(settings)
    env = environment(game)
    obs = env.reset()
    taken = 0
    for taken in range(1, steps + 1):  # noqa: B007 (used after loop)
        obs = env.step(agent.decide(obs))
        if obs.state is EnvironmentState.WIN:
            break
    chat = agent.chat
    return {
        "levels": obs.levels_completed,
        "actions": taken,
        "model_calls": agent.model_calls,
        "in_tok": chat.input_tokens,
        "out_tok": chat.output_tokens,
        "reasoning_tok": chat.reasoning_tokens,
    }


def play_harness(arm: str, game: str, steps: int) -> dict[str, object]:
    backend = OpenRouterBackend(
        OpenRouterConfig(model_name=MODEL, max_new_tokens=2048),
        os.environ["OPENROUTER_API_KEY"],
    )
    model = (
        ActionOnlyModelAdapter(backend)
        if arm == "action_only"
        else StructuredModelAdapter(backend)
    )
    limits = {
        Resource.ACTIONS: steps,
        Resource.MODEL_CALLS: steps * 3,
        Resource.INPUT_TOKENS: 20_000_000,
        Resource.OUTPUT_TOKENS: 5_000_000,
    }
    config = RunConfig(
        run_id=f"{arm}-{game}-{int(time.time())}",
        experiment_id="duck-vs-harness",
        game_id=game,
        max_turns=steps * 3,
        max_consecutive_decision_failures=5,
        output_dir=OUT / "traces",
        budget=BudgetConfig(limits=limits),
    )
    result = build_runner(config, environment(game), model).run()
    m = result.metrics
    return {
        "levels": m.levels_completed,
        "actions": m.environment_actions,
        "model_calls": m.model_calls,
        "in_tok": m.input_tokens,
        "out_tok": m.output_tokens,
        "stop": result.stop_reason,
    }


def main(arm: str, steps: int, games: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for game in games:
        started = time.time()
        try:
            row = (
                play_duck(game, steps)
                if arm == "duck"
                else play_harness(arm, game, steps)
            )
        except Exception as error:  # keep the other games running
            traceback.print_exc()
            row = {"error": f"{type(error).__name__}: {error}"[:300]}
        row |= {"arm": arm, "game": game, "secs": round(time.time() - started)}
        print(json.dumps(row), flush=True)
        (OUT / f"{arm}-{game}.json").write_text(json.dumps(row))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
