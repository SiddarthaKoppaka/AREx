"""Exact cell-level differences between two frames (single implementation)."""

from collections import Counter

from arc_agi_3.context.frame_components import FrameLike
from arc_agi_3.contracts.transition import CellChange, ChangedRegion, ValueCountChange

Cell = tuple[int, int, int]


def frame_cells(frame: FrameLike) -> dict[Cell, int]:
    return {
        (layer_index, row_index, column_index): value
        for layer_index, layer in enumerate(frame)
        for row_index, row in enumerate(layer)
        for column_index, value in enumerate(row)
    }


def cell_changes(before: FrameLike, after: FrameLike) -> list[CellChange]:
    old, new = frame_cells(before), frame_cells(after)
    changes: list[CellChange] = []
    for key in sorted(old.keys() | new.keys()):
        if old.get(key) != new.get(key):
            z, y, x = key
            changes.append(
                CellChange(
                    layer=z, row=y, col=x, before=old.get(key), after=new.get(key)
                )
            )
    return changes


def value_count_changes(
    before: FrameLike, after: FrameLike
) -> tuple[ValueCountChange, ...]:
    old = Counter(frame_cells(before).values())
    new = Counter(frame_cells(after).values())
    return tuple(
        ValueCountChange(value=value, before=old[value], after=new[value])
        for value in sorted(old.keys() | new.keys())
        if old[value] != new[value]
    )


def changed_regions(changes: list[CellChange]) -> tuple[ChangedRegion, ...]:
    by_layer: dict[int, list[CellChange]] = {}
    for change in changes:
        by_layer.setdefault(change.layer, []).append(change)
    return tuple(
        ChangedRegion(
            layer=layer,
            min_row=min(item.row for item in items),
            min_col=min(item.col for item in items),
            max_row=max(item.row for item in items),
            max_col=max(item.col for item in items),
            changed_cells=len(items),
        )
        for layer, items in sorted(by_layer.items())
    )


def shape(frame: FrameLike) -> list[list[int]]:
    return [[len(row) for row in layer] for layer in frame]
