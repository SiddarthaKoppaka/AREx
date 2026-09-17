"""Small deterministic adapters for end-to-end golden traces."""

from pydantic import JsonValue

from arc_agi_3.contracts.decision import (
    AgentContext,
    CognitiveDecision,
    ExpectedOutcome,
    ModelResponse,
)
from arc_agi_3.contracts.enums import DecisionMode, EnvironmentState
from arc_agi_3.contracts.observation import Action, Observation


class FakeLineEnvironment:
    def __init__(self, target: int = 2) -> None:
        if target < 1:
            raise ValueError("target must be positive")
        self.position = 0
        self.target = target

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "fake-line", "version": "1", "target": self.target}

    def _observation(self) -> Observation:
        row = [0] * (self.target + 1)
        row[self.position] = 2
        won = self.position == self.target
        return Observation.build(
            game_id="fake-line",
            frame=[[row]],
            state=EnvironmentState.WIN if won else EnvironmentState.NOT_FINISHED,
            levels_completed=int(won),
            win_levels=1,
            available_actions=[] if won else [1],
            guid="fake-guid",
            full_reset=self.position == 0,
        )

    def reset(self) -> Observation:
        self.position = 0
        return self._observation()

    def step(self, action: Action) -> Observation:
        if action.action_id != 1:
            raise ValueError("fake-line only supports action 1")
        self.position = min(self.target, self.position + 1)
        return self._observation()

    def checkpoint(self) -> dict[str, JsonValue]:
        return {"position": self.position}

    def restore(self, state: dict[str, JsonValue]) -> Observation:
        position = state.get("position")
        if not isinstance(position, int):
            raise ValueError("invalid fake environment checkpoint")
        self.position = position
        return self._observation()


class ScriptedModel:
    def __init__(self, decisions: list[CognitiveDecision]) -> None:
        self.decisions = list(decisions)

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "scripted", "version": "1"}

    def decide(self, context: AgentContext) -> ModelResponse:
        if not self.decisions:
            raise RuntimeError("scripted model has no decision remaining")
        return ModelResponse(decision=self.decisions.pop(0))


def successful_script() -> list[CognitiveDecision]:
    first = CognitiveDecision(
        assessment="The scripted fixture authorizes one move.",
        intent="Advance once and inspect the result.",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        expected_outcome=ExpectedOutcome(state=EnvironmentState.NOT_FINISHED),
    )
    second = CognitiveDecision(
        assessment="The scripted fixture authorizes the final move.",
        intent="Advance to the terminal state.",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        expected_outcome=ExpectedOutcome(
            state=EnvironmentState.WIN, min_levels_completed=1
        ),
    )
    return [first, second]
