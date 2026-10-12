"""JevBackend: fails loudly on missing credentials, never crashes on a
bad/slow response, never retries an LLM-repair loop against a typed API."""

import json
import urllib.error

import pytest

from arc_agi_3.adapters.jev_backend import JevBackend, JevBackendError
from arc_agi_3.contracts.system1 import System1DecisionRequest
from tests.system1_fixtures import two_candidate_request


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
    monkeypatch.setattr(
        backend,
        "_post",
        lambda _req: {
            "trial_id": "trial-1",
            "selected_candidate_id": "move-down",
            "candidate_probabilities": [],
        },
    )
    result = backend.decide(request())
    assert result.selected_candidate_id == "move-down"
    assert result.latency_ms is not None


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
