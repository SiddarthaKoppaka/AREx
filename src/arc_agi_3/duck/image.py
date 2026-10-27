"""Render a grid as a PNG using the official ARC 16-color palette."""

import io
from collections.abc import Sequence
from typing import Any

PALETTE = [
    (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
    (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
    (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
]  # fmt: skip


def render_image(grid: Sequence[Sequence[int]], scale: int = 8) -> Any:
    from PIL import Image

    image = Image.new("RGB", (len(grid[0]), len(grid)))
    image.putdata([PALETTE[cell % 16] for row in grid for cell in row])
    return image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )


def render_png(grid: Sequence[Sequence[int]], scale: int = 8) -> bytes:
    buffer = io.BytesIO()
    render_image(grid, scale).save(buffer, format="PNG")
    return buffer.getvalue()
