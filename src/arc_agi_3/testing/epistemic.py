"""Game-general fixtures for epistemic-loop tests.

The shift environment has no goal semantics: ACTION1 translates the value-3
component one column right, ACTION2 changes nothing. Tests must not encode
this in model logic; reactive models read only generic context fields.
"""

from collections.abc import Callable, Sequence

from pydantic import JsonValue

from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision, ModelResponse
from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.contracts.observation import Action, Observation


def grid(
    rows: Sequence[Sequence[int]], *, actions: Sequence[int] = (1, 2)
) -> Observation:
    return Observation.build(
        game_id="shift",
        frame=[rows],
        state=EnvironmentState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=actions,
    )


class FakeShiftEnvironment:
    def __init__(self, size: int = 8) -> None:
        self.size, self.column = size, 1

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "fake-shift", "version": "1", "restore_supported": True}

    def _observation(self) -> Observation:
        rows = [[0] * self.size for _ in range(self.size)]
        rows[5][5] = 1
        rows[2][self.column] = rows[2][self.column + 1] = 3
        return grid(rows)

    def reset(self) -> Observation:
        self.column = 1
        return self._observation()

    def step(self, action: Action) -> Observation:
        if action.action_id == 1:
            self.column = min(self.size - 2, self.column + 1)
        return self._observation()

    def checkpoint(self) -> dict[str, JsonValue]:
        return {"column": self.column}

    def restore(self, state: dict[str, JsonValue]) -> Observation:
        column = state.get("column")
        if not isinstance(column, int):
            raise ValueError("invalid fake-shift checkpoint")
        self.column = column
        return self._observation()


Policy = Callable[[AgentContext], CognitiveDecision]


class ReactiveModel:
    """Test double for the LM: one policy per turn, every context captured."""

    def __init__(self, policies: list[Policy]) -> None:
        self.policies = list(policies)
        self.contexts: list[AgentContext] = []

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "reactive", "version": "1"}

    def decide(self, context: AgentContext) -> ModelResponse:
        self.contexts.append(context)
        if not self.policies:
            raise RuntimeError("reactive model has no policy remaining")
        return ModelResponse(decision=self.policies.pop(0)(context))
