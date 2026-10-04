"""Exact frame cropping, shared by both region-view contracts.

`EvidenceQuery.region` (x/y/width/height, for `retrieve_evidence`) and
`FrameRegionRequest` (min_row/max_row/min_col/max_col, for the dedicated
`inspect_frame_region` tool) describe the same crop differently; both call
`crop_region` so the bound-checking logic exists exactly once.
"""

from typing import Any

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.evidence import FrameRegionRequest


def crop_region(
    frame: list[Any], layer: int, x: int, y: int, width: int, height: int
) -> list[list[int]]:
    if layer >= len(frame):
        raise ValueError("region layer is outside the frame")
    grid = frame[layer]
    rows = grid[y : y + height]
    if y + height > len(grid) or any(x + width > len(row) for row in rows):
        raise ValueError("region is outside the frame")
    return [list(row[x : x + width]) for row in rows]


def region_view(event: EventEnvelope, request: FrameRegionRequest) -> dict[str, Any]:
    frame = event.payload.get("frame")
    if not isinstance(frame, list):
        raise ValueError("requested event has no frame")
    cells = crop_region(
        frame,
        request.layer,
        request.min_col,
        request.min_row,
        request.max_col - request.min_col + 1,
        request.max_row - request.min_row + 1,
    )
    return {
        "event_id": event.event_id,
        "layer": request.layer,
        "min_row": request.min_row,
        "max_row": request.max_row,
        "min_col": request.min_col,
        "max_col": request.max_col,
        "cells": cells,
    }
