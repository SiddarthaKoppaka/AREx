"""Replaceable environment and inference boundaries."""

from .ollama import OllamaBackend, OllamaConfig
from .protocols import EnvironmentAdapter, ModelAdapter
from .transformers import TransformersBackend, TransformersConfig

__all__ = [
    "EnvironmentAdapter",
    "ModelAdapter",
    "OllamaBackend",
    "OllamaConfig",
    "TransformersBackend",
    "TransformersConfig",
]
