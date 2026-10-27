"""Text views of a grid: ASCII rendering and cell-level diffs."""

from collections.abc import Sequence

Grid = Sequence[Sequence[int]]
Change = tuple[int, int, int, int]
SYMBOLS = "0123456789ABCDEF"


def ascii_grid(grid: Grid) -> str:
    """One character per cell, one line per row (hex digit = color index)."""
    return "\n".join("".join(SYMBOLS[cell % 16] for cell in row) for row in grid)


def changes(prev: Grid | None, cur: Grid, limit: int = 20) -> tuple[int, list[Change]]:
    """Return (total changed cells, first `limit` as (row, col, old, new))."""
    if prev is None or len(prev) != len(cur) or len(prev[0]) != len(cur[0]):
        return 0, []
    found = [
        (r, c, prev[r][c], new)
        for r, row in enumerate(cur)
        for c, new in enumerate(row)
        if prev[r][c] != new
    ]
    return len(found), found[:limit]


def describe_diff(prev: Grid | None, cur: Grid) -> str:
    """Changed cells grouped by (old -> new) color, with each group's bounding box."""
    if prev is None:
        return "no previous frame"
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c, old, new in changes(prev, cur, limit=len(cur) * len(cur[0]))[1]:
        groups.setdefault((old, new), []).append((r, c))
    if not groups:
        return "no cells changed"
    return "\n".join(
        f"{old:X}->{new:X}: {len(cells)} cells, rows {min(r for r, _ in cells)}-"
        f"{max(r for r, _ in cells)}, cols {min(c for _, c in cells)}-"
        f"{max(c for _, c in cells)}"
        for (old, new), cells in sorted(groups.items())
    )
