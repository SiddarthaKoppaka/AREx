"""Chat backend for a local OpenAI-compatible server (SGLang or vLLM)."""

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .chat import Message, latest_image_only, strip_think


def openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for m in latest_image_only(messages):
        content: Any = m["content"]
        if "image" in m:
            url = "data:image/png;base64," + base64.b64encode(m["image"]).decode()
            content = [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": m["content"]},
            ]
        converted.append({"role": m["role"], "content": content})
    return converted


class OpenAICompatChat:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "duck",
        *,
        max_tokens: int = 2048,
        timeout: float = 300,
        thinking: bool = False,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        retries: int = 3,
    ) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.max_tokens, self.timeout, self.thinking = max_tokens, timeout, thinking
        self.api_key, self.retries = api_key, retries
        self.reasoning_effort = reasoning_effort
        self.input_tokens = self.output_tokens = self.reasoning_tokens = 0

    def payload(self, messages: list[Message]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages(messages),
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def _request(self, messages: list[Message]) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(self.payload(messages)).encode(),
            headers=headers,
        )

    def _post(self, request: urllib.request.Request) -> dict[str, Any]:
        """Fetch with a hard wall-clock limit (urlopen's timeout is per read)."""
        result: list[Any] = []

        def fetch() -> None:
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                    result.append(json.load(reply))
            except BaseException as error:  # handed back to the caller's thread
                result.append(error)

        worker = threading.Thread(target=fetch, daemon=True)
        worker.start()
        worker.join(self.timeout)
        if not result:
            raise TimeoutError(f"no complete reply within {self.timeout}s")
        if isinstance(result[0], BaseException):
            raise result[0]
        return result[0]  # type: ignore[no-any-return]

    def chat(self, messages: list[Message]) -> str:
        for attempt in range(self.retries + 1):
            try:
                body = self._post(self._request(messages))
                usage = body.get("usage") or {}
                self.input_tokens += usage.get("prompt_tokens", 0)
                self.output_tokens += usage.get("completion_tokens", 0)
                details = usage.get("completion_tokens_details") or {}
                self.reasoning_tokens += details.get("reasoning_tokens", 0)
                return strip_think(body["choices"][0]["message"]["content"] or "")
            except OSError as error:  # HTTPError, URLError, SSLError, timeouts
                code = getattr(error, "code", None)
                if code is not None and code not in (429, 500, 502, 503):
                    raise
                if attempt == self.retries:
                    raise
                headers = getattr(error, "headers", None)
                wait = headers.get("Retry-After", "") if headers else ""
                time.sleep(min(float(wait) if wait.isdigit() else 5 * 2**attempt, 60))
        raise AssertionError("unreachable")
