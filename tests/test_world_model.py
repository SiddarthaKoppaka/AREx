"""Declarative models are versioned, executable, and replayable."""

import pytest

from arc_agi_3.contracts.observation import Action
from arc_agi_3.world_model import (
    DeclarativeRule,
    SimulationRequest,
    SymbolicState,
    WorldModel,
    WorldModelRuntime,
    WorldModelStore,
)
from arc_agi_3.world_model.contracts import ReplayCase


def line_model(version: int = 1) -> WorldModel:
    return WorldModel(
        model_id="line",
        version=version,
        rules=(
            DeclarativeRule(
                rule_id="0-1",
                action=Action(action_id=1),
                preconditions={"position": 0},
                effects={"position": 1},
            ),
            DeclarativeRule(
                rule_id="1-2",
                action=Action(action_id=1),
                preconditions={"position": 1},
                effects={"position": 2},
            ),
        ),
    )


def test_simulation_and_historical_replay() -> None:
    runtime = WorldModelRuntime(line_model())
    start = SymbolicState.build({"position": 0})
    middle = SymbolicState.build({"position": 1})
    request = SimulationRequest(
        model_id="line",
        model_version=1,
        initial_state=start,
        actions=(Action(action_id=1), Action(action_id=1)),
    )
    result = runtime.simulate(request)
    assert result.status == "complete"
    assert result.final_state.facts == {"position": 2}
    replay = runtime.replay(
        (
            ReplayCase(
                case_id="known",
                before=start,
                action=Action(action_id=1),
                expected_after=middle,
            ),
        )
    )
    assert replay[0].passed


def test_invalid_transition_is_evidence_not_invented_behavior() -> None:
    runtime = WorldModelRuntime(line_model())
    request = SimulationRequest(
        model_id="line",
        model_version=1,
        initial_state=SymbolicState.build({"position": 2}),
        actions=(Action(action_id=1),),
    )
    assert runtime.simulate(request).status == "failed"


def test_store_requires_monotonic_versions() -> None:
    store = WorldModelStore()
    store.add_many((line_model(),))
    with pytest.raises(ValueError, match="version 2"):
        store.add_many((line_model(),))
    store.add_many((line_model(2),))
    assert len(store.history("line")) == 2
