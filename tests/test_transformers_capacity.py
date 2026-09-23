"""The loaded model can lower the configured context ceiling."""

from types import SimpleNamespace

import pytest

from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig
from arc_agi_3.adapters.transformers_limits import InputTokenLimitError


class TokenIds(list[int]):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self))


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return {"input_ids": TokenIds([1, 2, 3])}


class ShortContextModel:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=514)

    def __init__(self) -> None:
        self.generated = False

    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        self.generated = True
        return [[1, 2, 3]]


def test_backend_uses_loaded_models_actual_context_capacity() -> None:
    model, tokenizer = ShortContextModel(), Tokenizer()
    config = TransformersConfig(
        model_path="/weights",
        max_context_tokens=32768,
        max_input_tokens=30720,
        max_new_tokens=2,
        template_and_generation_margin=510,
    )
    backend = TransformersBackend(config, model=model, tokenizer=tokenizer)
    assert backend.effective_max_input_tokens == 2
    with pytest.raises(InputTokenLimitError):
        backend.generate("choose", {"type": "object"})
    assert model.generated is False
