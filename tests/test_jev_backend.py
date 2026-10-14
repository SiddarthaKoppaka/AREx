"""JevBackend: fails loudly on missing credentials, never crashes on a
bad/slow response, never retries an LLM-repair loop against a typed API.
Mocks TypeSafe's real /v1/systemone wire shape, not our internal contract."""

import json
import urllib.error

import pytest

from arc_agi_3.adapters.jev_backend import JevBackend, JevBackendError
from arc_agi_3.contracts.system1 import System1DecisionRequest
from tests.system1_fixtures import two_candidate_request, typesafe_response


def request() -> System1DecisionRequest:
    return System1DecisionRequest.model_validate(two_candidate_request())


def test_missing_api_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(JevBackendError, match="JEV_API_KEY"):
        JevBackend("https://jev.example")


def test_api_key_is_never_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "super-secret")
    backend = JevBackend("https://jev.example")
    assert "super-secret" not in json.dumps(backend.metadata)


def test_successful_decision_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "k")
    backend = JevBackend("https://jev.example")
    monkeypatch.setattr(backend, "_post", lambda _req: typesafe_response("move-down"))
    result = backend.decide(request())
    assert result.selected_candidate_id == "move-down"
    assert result.confidence == 0.84
    assert result.stalled_probability == 0.91
    assert {p.candidate_id: p.probability for p in result.candidate_probabilities} == {
        "inspect-region": 0.4,
        "move-down": 0.6,
    }
    assert result.latency_ms is not None


def test_outgoing_request_uses_the_real_choice_question_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JEV_API_KEY", "k")
    backend = JevBackend("https://jev.example", model="jev-test")
    captured: dict[str, object] = {}

    def _capture(req: System1DecisionRequest) -> dict[str, object]:
        from arc_agi_3.adapters.jev_wire import build_request_body

        captured.update(build_request_body(backend._model, req))
        return typesafe_response("move-down")

    monkeypatch.setattr(backend, "_post", _capture)
    backend.decide(request())
    assert captured["model"] == "jev-test"
    question = captured["questions"]["select_candidate"]
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "inspect-region": "Inspect the collision region.",
        "move-down": "Move down to test directionality.",
    }


def test_malformed_response_raises_jev_error_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JEV_API_KEY", "k")
    backend = JevBackend("https://jev.example")
    monkeypatch.setattr(backend, "_post", lambda _req: {"not": "a valid result"})
    with pytest.raises(JevBackendError, match="invalid decision"):
        backend.decide(request())


def test_network_failure_raises_jev_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "k")
    backend = JevBackend("https://jev.example")

    def _raise(_req: System1DecisionRequest) -> dict[str, object]:
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(backend, "_post", _raise)
    with pytest.raises(JevBackendError, match="JEV request failed"):
        backend.decide(request())
