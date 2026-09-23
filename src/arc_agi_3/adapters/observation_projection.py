"""Lossless but compact current observation for the model prompt."""

from pydantic import JsonValue

from arc_agi_3.context.frame_views import frame_summary, rle_frame
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.trace.canonical import canonical_json


def observation_view(observation: Observation) -> dict[str, JsonValue]:
    values = observation.model_dump(mode="json", exclude={"frame"})
    frame = observation.frame
    encoded = rle_frame(frame)
    raw = [[list(row) for row in layer] for layer in frame]
    if len(canonical_json(encoded)) < len(canonical_json(raw)):
        values["frame_view"] = encoded
    else:
        values["frame_view"] = {"encoding": "raw", "layers": raw}
    values["mechanical_summary"] = frame_summary(frame)
    return values
