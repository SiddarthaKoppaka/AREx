"""Adapter for TypeSafe's real JEV System-1 API. See ADR-0010.

Verified against https://api.typesafe.ai/openapi.json on 2026-10-14 -
POST /v1/systemone, bearer auth. jev_wire.py owns the translation to/from
System1DecisionRequest/Result; this module owns the HTTP call, the
credential, and turning any failure into a clean, non-crashing error.
Never logs or traces the API key.
"""

import json
import time
import urllib.error
import urllib.request
from os import environ
from typing import Any

from pydantic import JsonValue

from arc_agi_3.contracts.system1 import System1DecisionRequest, System1DecisionResult

from .jev_wire import build_request_body, parse_response_body

JEV_API_KEY_ENV = "JEV_API_KEY"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_SECONDS = 10.0


class JevBackendError(ValueError):
    """The JEV API failed, timed out, or returned an unusable response.

    A ValueError so the existing run_tool try/except turns this into a
    clean failed ToolResult rather than crashing the episode.
    """


class JevBackend:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        api_key_env: str = JEV_API_KEY_ENV,
    ) -> None:
        api_key = environ.get(api_key_env)
        if not api_key:
            raise JevBackendError(
                f"JEV is configured but {api_key_env} is not set in the environment"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "jev", "base_url": self._base_url, "model": self._model}

    def decide(self, request: System1DecisionRequest) -> System1DecisionResult:
        started = time.monotonic()
        try:
            raw = self._post(request)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise JevBackendError(f"JEV request failed: {error}") from error
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            result = parse_response_body(request, raw)
        except (KeyError, TypeError, ValueError) as error:
            raise JevBackendError(
                f"JEV returned an invalid decision: {error}"
            ) from error
        if result.latency_ms is None:
            result = result.model_copy(update={"latency_ms": latency_ms})
        return result

    def _post(self, request: System1DecisionRequest) -> dict[str, Any]:
        body = json.dumps(build_request_body(self._model, request)).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self._base_url}/v1/systemone",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(
            http_request, timeout=self._timeout_seconds
        ) as response:
            parsed: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return parsed
