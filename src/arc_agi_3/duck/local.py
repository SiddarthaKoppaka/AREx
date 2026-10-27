"""Local episode loop for the Duck agent."""

from arc_agi_3.adapters.protocols import EnvironmentAdapter
from arc_agi_3.contracts.enums import EnvironmentState

from .agent import DuckAgent


def run_episode(env: EnvironmentAdapter, agent: DuckAgent, max_steps: int) -> int:
    """Play until WIN or `max_steps` actions; return levels completed."""
    obs = env.reset()
    for _ in range(max_steps):
        if obs.state is EnvironmentState.WIN:
            break
        obs = env.step(agent.decide(obs))
    return obs.levels_completed
