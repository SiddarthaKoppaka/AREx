"""Replaceable environment and inference boundaries."""

from .ollama import OllamaBackend, OllamaConfig
from .protocols import EnvironmentAdapter, ModelAdapter

__all__ = ["EnvironmentAdapter", "ModelAdapter", "OllamaBackend", "OllamaConfig"]
