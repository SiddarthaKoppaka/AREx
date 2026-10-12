"""Adapter for the JEV System-1 decision API. See ADR-0010.

The wire format assumed here is the System1DecisionRequest/Result JSON
shape itself, since no JEV API documentation was available while writing
this adapter - replace `_post` with the real client call once JEV's
actual request/response schema is known. Never logs or traces the API key.
"""

import json
import time
import urllib.error
import urllib.request
from os import environ
from typing import Any

from pydantic import JsonValue

from arc_agi_3.contracts.system1 import System1DecisionRequest, System1DecisionResult
from arc_agi_3.trace.canonical import canonical_json

JEV_API_KEY_ENV = "JEV_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 10.0


class JevBackendError(ValueError):
    """The JEV API failed, timed out, or returned an unusable response.

    A ValueError so the existing run_tool try/except turns this into a
    clean failed ToolResult rather than crashing the episode.
    """


class JevBackend:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        api_key_env: str = JEV_API_KEY_ENV,
    ) -> None:
        api_key = environ.get(api_key_env)
        if not api_key:
            raise JevBackendError(
                f"JEV is configured but {api_key_env} is not set in the environment"
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "jev", "base_url": self._base_url}

    def decide(self, request: System1DecisionRequest) -> System1DecisionResult:
        started = time.monotonic()
        try:
            raw = self._post(request)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise JevBackendError(f"JEV request failed: {error}") from error
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            result = System1DecisionResult.model_validate(raw)
        except Exception as error:
            raise JevBackendError(
                f"JEV returned an invalid decision: {error}"
            ) from error
        if result.latency_ms is None:
            result = result.model_copy(update={"latency_ms": latency_ms})
        return result

    def _post(self, request: System1DecisionRequest) -> dict[str, Any]:
        body = canonical_json(request).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self._base_url}/v1/decide",
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
