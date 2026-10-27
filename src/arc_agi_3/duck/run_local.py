"""Play one bundled game locally: `python -m arc_agi_3.duck.run_local ls20 40`."""

import sys
import tempfile
from pathlib import Path

from arc_agi_3.adapters.arc import ArcEnvironmentAdapter
from arc_agi_3.contracts.enums import EnvironmentState

from .factory import make_agent
from .settings import DuckSettings

ENV_ROOT = Path("arc-prize-2026-arc-agi-3/environment_files")


def main(game: str = "ls20", steps: int = 40) -> int:
    import arc_agi

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ENV_ROOT),
        recordings_dir=tempfile.mkdtemp(),
    )
    env = ArcEnvironmentAdapter(arcade.make(game, seed=0, save_recording=False))
    agent = make_agent(DuckSettings.from_env())
    obs = env.reset()
    for step in range(1, steps + 1):
        action = agent.decide(obs)
        obs = env.step(action)
        print(
            f"step {step:3d} action={action.action_id} {dict(action.data)} "
            f"state={obs.state.value} levels={obs.levels_completed}/{obs.win_levels}",
            flush=True,
        )
        if obs.state is EnvironmentState.WIN:
            break
    print(f"model_calls={agent.model_calls} fallbacks={agent.fallbacks}")
    return obs.levels_completed


if __name__ == "__main__":
    main(*(int(a) if a.isdigit() else a for a in sys.argv[1:]))  # type: ignore[arg-type]
