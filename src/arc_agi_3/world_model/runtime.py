"""Bounded interpreter for declarative rules; no arbitrary code execution."""

from typing import Literal

from arc_agi_3.contracts.observation import Action

from .contracts import (
    DeclarativeRule,
    ReplayCase,
    ReplayCheck,
    SimulationRequest,
    SimulationResult,
    SimulationStep,
    SymbolicState,
    WorldModel,
)


class WorldModelRuntime:
    def __init__(self, model: WorldModel) -> None:
        self.model = model

    @staticmethod
    def _matches(state: SymbolicState, rule: DeclarativeRule) -> bool:
        return all(
            state.facts.get(key) == value for key, value in rule.preconditions.items()
        )

    def transitions(
        self, state: SymbolicState
    ) -> tuple[tuple[DeclarativeRule, SymbolicState], ...]:
        results = []
        for rule in sorted(self.model.rules, key=lambda item: item.rule_id):
            if self._matches(state, rule):
                facts = {**state.facts, **rule.effects}
                results.append((rule, SymbolicState.build(facts)))
        return tuple(results)

    def apply(
        self, state: SymbolicState, action: Action
    ) -> tuple[
        Literal["applied", "invalid", "ambiguous"],
        SymbolicState | None,
        str | None,
    ]:
        matches = [
            (rule, after)
            for rule, after in self.transitions(state)
            if rule.action == action
        ]
        if not matches:
            return "invalid", None, None
        if len(matches) > 1:
            return "ambiguous", None, None
        rule, after = matches[0]
        return "applied", after, rule.rule_id

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        if (request.model_id, request.model_version) != (
            self.model.model_id,
            self.model.version,
        ):
            raise ValueError("simulation request targets a different model version")
        current = request.initial_state
        steps: list[SimulationStep] = []
        for index, action in enumerate(request.actions):
            status, after, rule_id = self.apply(current, action)
            steps.append(
                SimulationStep(
                    index=index,
                    action=action,
                    before=current,
                    after=after,
                    rule_id=rule_id,
                    status=status,
                )
            )
            if after is None:
                return SimulationResult(
                    model_id=self.model.model_id,
                    model_version=self.model.version,
                    status="partial" if index else "failed",
                    steps=tuple(steps),
                    final_state=current,
                )
            current = after
        return SimulationResult(
            model_id=self.model.model_id,
            model_version=self.model.version,
            status="complete",
            steps=tuple(steps),
            final_state=current,
        )

    def replay(self, cases: tuple[ReplayCase, ...]) -> tuple[ReplayCheck, ...]:
        from .replay import replay_cases

        return replay_cases(self, cases)
