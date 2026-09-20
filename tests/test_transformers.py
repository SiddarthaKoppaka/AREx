"""Attached-weight inference stays offline and behind the backend protocol."""

from types import SimpleNamespace
from typing import Any

import pytest

from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig
from arc_agi_3.adapters.transformers_limits import InputTokenLimitError


class TokenIds(list[int]):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self))


class Tokenizer:
    def __init__(self) -> None:
        self.template: dict[str, Any] = {}

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.template = {"messages": messages, **kwargs}
        return {"input_ids": TokenIds([10, 11, 12])}

    def decode(self, tokens, **kwargs):  # type: ignore[no-untyped-def]
        assert tokens == [20, 21]
        assert kwargs == {"skip_special_tokens": True}
        return '{"mode":"stop"}'


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}

    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        self.arguments = kwargs
        return [[10, 11, 12, 20, 21]]

    def eval(self):  # type: ignore[no-untyped-def]
        return self


def test_backend_generates_direct_structured_output() -> None:
    model, tokenizer = Model(), Tokenizer()
    config = TransformersConfig(model_path="/kaggle/input/qwen", max_new_tokens=7)
    backend = TransformersBackend(config, model=model, tokenizer=tokenizer)
    generated = backend.generate("choose", {"type": "object"})
    assert "JSON_SCHEMA" in tokenizer.template["messages"][0]["content"]
    assert tokenizer.template["enable_thinking"] is False
    assert model.arguments["do_sample"] is False
    assert model.arguments["max_new_tokens"] == 7
    assert model.arguments["max_time"] == 180
    assert generated.text == '{"mode":"stop"}'
    assert generated.usage.input_tokens == 3
    assert generated.usage.output_tokens == 2
    assert backend.metadata["provider"] == "transformers"


def test_input_token_guard_rejects_complete_prompt_without_generating() -> None:
    model, tokenizer = Model(), Tokenizer()
    config = TransformersConfig(
        model_path="/weights", model_name="example", max_input_tokens=2
    )
    backend = TransformersBackend(config, model=model, tokenizer=tokenizer)
    with pytest.raises(InputTokenLimitError) as caught:
        backend.generate("choose", {"type": "object"})
    assert (caught.value.actual, caught.value.limit, caught.value.model) == (
        3,
        2,
        "example",
    )
    assert "JSON_SCHEMA" in tokenizer.template["messages"][0]["content"]
    assert model.arguments == {}


def test_backend_loads_text_only_qwen_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    model, tokenizer = Model(), Tokenizer()
    tokenizer_factory = SimpleNamespace(from_pretrained=lambda *args, **kw: tokenizer)
    model_factory = SimpleNamespace(from_pretrained=lambda *args, **kw: model)
    module = SimpleNamespace(
        AutoTokenizer=tokenizer_factory,
        AutoModelForCausalLM=model_factory,
    )
    monkeypatch.setattr(
        "arc_agi_3.adapters.transformers_loader.import_module", lambda name: module
    )
    backend = TransformersBackend(TransformersConfig(model_path="/weights"))
    assert backend.model is model
    assert backend.tokenizer is tokenizer
