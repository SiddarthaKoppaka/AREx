"""Offline Hugging Face chat backend; text-only or vision via a processor."""

import io
from typing import Any

from arc_agi_3.adapters.model_invocation import run_model

from .chat import Message, latest_image_only, strip_think


def hf_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert to the multimodal chat format (image part before its text)."""
    from PIL import Image

    converted: list[dict[str, Any]] = []
    for m in latest_image_only(messages):
        content: Any = m["content"]
        if "image" in m:
            picture = Image.open(io.BytesIO(m["image"])).convert("RGB")
            content = [
                {"type": "image", "image": picture},
                {"type": "text", "text": m["content"]},
            ]
        converted.append({"role": m["role"], "content": content})
    return converted


class HFChatBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        processor: Any | None = None,
        max_new_tokens: int = 2048,
        max_time_seconds: float = 300,
        enable_thinking: bool = False,
    ) -> None:
        self.model, self.tokenizer, self.processor = model, tokenizer, processor
        self.max_new_tokens, self.max_time = max_new_tokens, max_time_seconds
        self.enable_thinking = enable_thinking

    def chat(self, messages: list[Message]) -> str:
        owner = self.processor or self.tokenizer
        prepared = hf_messages(messages) if self.processor else messages
        inputs = owner.apply_chat_template(
            prepared,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)
        count = int(inputs["input_ids"].shape[-1])
        output, _ = run_model(
            self.model,
            inputs,
            count,
            max_new_tokens=self.max_new_tokens,
            max_time_seconds=self.max_time,
        )
        return strip_think(self.tokenizer.decode(output, skip_special_tokens=True))
