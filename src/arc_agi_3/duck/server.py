"""Launch a local SGLang/vLLM OpenAI-compatible server inside the notebook."""

import os
import subprocess
import sys
import time
import urllib.request
from typing import Literal

Engine = Literal["sglang", "vllm"]
NAME = "duck"


def server_command(
    engine: Engine, model_path: str, *, port: int = 8000, tp: int = 1,
    context: int = 32768,
) -> list[str]:  # fmt: skip
    common = ["--port", str(port), "--served-model-name", NAME]
    if engine == "sglang":
        return [
            sys.executable, "-m", "sglang.launch_server", "--model-path", model_path,
            "--tp", str(tp), "--context-length", str(context), *common,
        ]  # fmt: skip
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path, "--tensor-parallel-size", str(tp),
        "--max-model-len", str(context), *common,
    ]  # fmt: skip


def launch_server(command: list[str]) -> subprocess.Popen[bytes]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    return subprocess.Popen(command, env=env)


def wait_ready(base_url: str, timeout: float = 1800, poll: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5):
                return
        except OSError:
            time.sleep(poll)
    raise TimeoutError(f"model server not ready at {base_url} after {timeout}s")
