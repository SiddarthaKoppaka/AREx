"""Compact-by-default current observation for the model prompt.

The exact grid is large (a full 64x64 frame dominates a cognitive prompt) and
is never required to pick the next action: `mechanical_summary` already has
dimensions, the value histogram, and the smallest connected components. The
raw frame stays exactly retrievable through `retrieve_evidence`/
`inspect_frame_region` by `current_observation_event_id`; set
`include_raw_frame=True` only when a caller explicitly wants it inlined.
"""

from pydantic import JsonValue

from arc_agi_3.context.frame_views import frame_summary, rle_frame
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.trace.canonical import canonical_json


def observation_view(
    observation: Observation, *, include_raw_frame: bool = False
) -> dict[str, JsonValue]:
    values = observation.model_dump(mode="json", exclude={"frame"})
    frame = observation.frame
    values["mechanical_summary"] = frame_summary(frame)
    if include_raw_frame:
        encoded = rle_frame(frame)
        raw = [[list(row) for row in layer] for layer in frame]
        if len(canonical_json(encoded)) < len(canonical_json(raw)):
            values["frame_view"] = encoded
        else:
            values["frame_view"] = {"encoding": "raw", "layers": raw}
    return values
