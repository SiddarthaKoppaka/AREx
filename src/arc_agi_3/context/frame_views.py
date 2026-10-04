"""Lossless frame encoding and mechanical frame statistics."""

from collections import Counter
from collections.abc import Sequence

from .frame_components import components

FrameLike = Sequence[Sequence[Sequence[int]]]


def rle_frame(frame: FrameLike) -> dict[str, object]:
    """Encode each row as value/count runs, preserving layers and dimensions."""
    layers: list[list[list[list[int]]]] = []
    for layer in frame:
        rows: list[list[list[int]]] = []
        for row in layer:
            runs: list[list[int]] = []
            for value in row:
                if runs and runs[-1][0] == value:
                    runs[-1][1] += 1
                else:
                    runs.append([value, 1])
            rows.append(runs)
        layers.append(rows)
    return {"encoding": "row_rle_v1", "layers": layers}


def decode_rle_frame(encoded: dict[str, object]) -> list[list[list[int]]]:
    """Reconstruct every layer and row, rejecting malformed or empty runs."""
    layers = encoded.get("layers")
    if encoded.get("encoding") != "row_rle_v1" or not isinstance(layers, list):
        raise ValueError("invalid row RLE encoding")
    decoded: list[list[list[int]]] = []
    for layer in layers:
        if not isinstance(layer, list):
            raise ValueError("invalid RLE layer")
        rows: list[list[int]] = []
        for runs in layer:
            if not isinstance(runs, list):
                raise ValueError("invalid RLE row")
            row: list[int] = []
            for run in runs:
                if (
                    not isinstance(run, list)
                    or len(run) != 2
                    or type(run[0]) is not int
                    or type(run[1]) is not int
                    or run[1] <= 0
                ):
                    raise ValueError("invalid RLE run")
                row.extend([run[0]] * run[1])
            rows.append(row)
        decoded.append(rows)
    return decoded


def frame_summary(frame: FrameLike) -> dict[str, object]:
    """Report sizes, values, and equal-value 4-connected regions only."""
    dimensions = [[len(layer), [len(row) for row in layer]] for layer in frame]
    histogram = Counter(value for layer in frame for row in layer for value in row)
    regions = components(frame)
    return {
        "dimensions": dimensions,
        "value_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "component_count": len(regions),
        "smallest_components": regions[:16],
        "omitted_component_count": max(0, len(regions) - 16),
    }
