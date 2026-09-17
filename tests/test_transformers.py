"""Attached-weight inference stays offline and behind the backend protocol."""

from typing import Any

from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig


class TokenIds(list[int]):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self))


class Processor:
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


def test_backend_generates_direct_structured_output() -> None:
    model, processor = Model(), Processor()
    config = TransformersConfig(model_path="/kaggle/input/qwen", max_new_tokens=7)
    backend = TransformersBackend(config, model=model, processor=processor)
    generated = backend.generate("choose", {"type": "object"})
    assert "JSON_SCHEMA" in processor.template["messages"][0]["content"]
    assert processor.template["enable_thinking"] is False
    assert model.arguments["do_sample"] is False
    assert model.arguments["max_new_tokens"] == 7
    assert generated.text == '{"mode":"stop"}'
    assert generated.usage.input_tokens == 3
    assert generated.usage.output_tokens == 2
    assert backend.metadata["provider"] == "transformers"
