"""Competition callback behavior without the official framework process."""

from pathlib import Path

import pytest

from arc_agi_3.audit import audit_agency_boundary
from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EnvironmentState, EventType
from arc_agi_3.contracts.execution import ActionChunk, ChunkStep
from arc_agi_3.contracts.observation import Action
from arc_agi_3.deployment.kaggle_bridge import SessionFinished
from arc_agi_3.testing import successful_script
from arc_agi_3.testing.callback import callback_bridge, callback_frame


def identity_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "arc_agi_3.deployment.kaggle_bridge.game_action_from_action",
        lambda value: value,
    )


def test_bridge_verifies_result_on_next_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_action(monkeypatch)
    bridge = callback_bridge(tmp_path, "callback-direct", successful_script())
    initial = callback_frame(0, EnvironmentState.NOT_FINISHED, [1])
    assert bridge.choose_action(initial).action_id == 1
    middle = callback_frame(1, EnvironmentState.NOT_FINISHED, [1])
    assert not bridge.is_done(middle)
    assert bridge.choose_action(middle).action_id == 1
    assert bridge.is_done(callback_frame(2, EnvironmentState.WIN, []))
    assert bridge.session.result is not None and bridge.session.result.metrics.succeeded
    assert audit_agency_boundary(bridge.session.io.events.read()) == ()


def test_chunk_continues_without_an_extra_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_action(monkeypatch)
    chunk = ActionChunk(
        chunk_id="callback-chunk",
        steps=(ChunkStep(action=Action(action_id=1)),) * 2,
    )
    decision = CognitiveDecision(
        assessment="authorize",
        intent="advance",
        mode=DecisionMode.EXECUTE,
        action_chunk=chunk,
    )
    bridge = callback_bridge(tmp_path, "callback-chunk", [decision])
    first = callback_frame(0, EnvironmentState.NOT_FINISHED, [1])
    second = callback_frame(1, EnvironmentState.NOT_FINISHED, [1])
    assert bridge.choose_action(first).action_id == 1
    assert bridge.choose_action(second).action_id == 1
    assert bridge.is_done(callback_frame(2, EnvironmentState.WIN, []))
    assert bridge.session.result is not None
    assert bridge.session.result.metrics.model_calls == 1


def test_game_over_remains_an_lm_decision_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_action(monkeypatch)
    reset = CognitiveDecision(
        assessment="level failed",
        intent="retry",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=0),
    )
    bridge = callback_bridge(tmp_path, "callback-reset", [reset])
    failed = callback_frame(0, EnvironmentState.GAME_OVER, [0])
    assert bridge.choose_action(failed).action_id == 0


def test_bridge_never_invents_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_action(monkeypatch)
    stop = CognitiveDecision(assessment="done", intent="stop", mode=DecisionMode.STOP)
    bridge = callback_bridge(tmp_path, "callback-stop", [stop])
    initial = callback_frame(0, EnvironmentState.NOT_FINISHED, [1])
    with pytest.raises(SessionFinished, match="will not invent"):
        bridge.choose_action(initial)
    kinds = {event.event_type for event in bridge.session.io.events.read()}
    assert EventType.ACTION not in kinds
