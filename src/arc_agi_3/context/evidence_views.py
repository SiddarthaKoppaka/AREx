"""Deterministic event views selected by the LM, with exact provenance."""

from typing import Any

from pydantic import TypeAdapter

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.evidence import EvidenceQuery
from arc_agi_3.contracts.observation import Frame

from .frame_delta import transition_delta
from .frame_views import frame_summary, rle_frame

_FRAME = TypeAdapter(Frame)


def view_event(
    event: EventEnvelope, history: list[EventEnvelope], query: EvidenceQuery
) -> dict[str, Any]:
    payload = event.payload
    frame = payload.get("frame")
    view = query.view
    if view == "full":
        content: object = payload
    elif view == "metadata":
        content = {key: value for key, value in payload.items() if key != "frame"}
    elif view == "summary":
        content = (
            frame_summary(_FRAME.validate_python(frame))
            if isinstance(frame, list)
            else _metadata(event)
        )
    elif view == "rle_frame":
        content = _frame_required(frame, rle_frame)
    elif view == "region":
        content = _region(frame, query)
    else:
        content = transition_delta(event, history)
    return {
        "event_id": event.event_id,
        "event_hash": event.event_hash,
        "event_type": event.event_type.value,
        "view": view,
        "content": content,
    }


def _metadata(event: EventEnvelope) -> dict[str, object]:
    return {key: value for key, value in event.payload.items() if key != "frame"}


def _frame_required(frame: object, encoder: Any) -> object:
    if not isinstance(frame, list):
        raise ValueError("requested event has no frame")
    return encoder(_FRAME.validate_python(frame))


def _region(frame: object, query: EvidenceQuery) -> dict[str, object]:
    if not isinstance(frame, list) or query.region is None:
        raise ValueError("requested event has no frame")
    region = query.region
    if region.layer >= len(frame):
        raise ValueError("region layer is outside the frame")
    layer = frame[region.layer]
    if region.y + region.height > len(layer) or any(
        region.x + region.width > len(row)
        for row in layer[region.y : region.y + region.height]
    ):
        raise ValueError("region is outside the frame")
    return {
        "layer": region.layer,
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
        "cells": [
            row[region.x : region.x + region.width]
            for row in layer[region.y : region.y + region.height]
        ],
    }
