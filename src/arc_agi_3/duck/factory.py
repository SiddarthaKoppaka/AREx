"""Build the chat backend and agent from settings (one model per process)."""

import os
from importlib import import_module

from .agent import DuckAgent
from .chat import ChatBackend
from .fit import preflight, total_vram_gb
from .hf_chat import HFChatBackend
from .openai_chat import OpenAICompatChat
from .server import NAME, Engine, launch_server, server_command, wait_ready
from .settings import DuckSettings

_chats: dict[DuckSettings, ChatBackend] = {}


def _hf_chat(settings: DuckSettings) -> ChatBackend:
    transformers = import_module("transformers")
    path = settings.model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = (
        transformers.AutoProcessor.from_pretrained(path, local_files_only=True)
        if settings.use_images
        else None
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, dtype="auto", device_map="auto", local_files_only=True
    ).eval()
    return HFChatBackend(
        model, tokenizer, processor=processor, enable_thinking=settings.thinking
    )


def _openrouter_chat(settings: DuckSettings) -> ChatBackend:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    return OpenAICompatChat(
        "https://openrouter.ai/api/v1",
        settings.model_name,
        api_key=key,
        thinking=settings.thinking,
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
    )


def _served_chat(settings: DuckSettings) -> ChatBackend:
    engine: Engine = "sglang" if settings.backend == "sglang" else "vllm"
    command = server_command(
        engine, settings.model_path, port=settings.port, tp=settings.tp,
        context=settings.context,
    )  # fmt: skip
    launch_server(command)
    base = f"http://127.0.0.1:{settings.port}/v1"
    wait_ready(base)
    return OpenAICompatChat(base, NAME, thinking=settings.thinking)


def build_chat(settings: DuckSettings) -> ChatBackend:
    preflight(settings.params_billion, settings.bits, total_vram_gb())
    if settings not in _chats:
        builders = {
            "hf": _hf_chat,
            "openrouter": _openrouter_chat,
            "sglang": _served_chat,
            "vllm": _served_chat,
        }
        builder = builders[settings.backend]
        _chats[settings] = builder(settings)
    return _chats[settings]


def make_agent(settings: DuckSettings, chat: ChatBackend | None = None) -> DuckAgent:
    """Fresh per-game agent state around the shared model."""
    return DuckAgent(
        chat or build_chat(settings),
        max_inner=settings.max_inner,
        use_images=settings.use_images,
        trace_path=settings.trace_path,
    )
