"""Generic grid analysis the model may call from its REPL."""

from collections import Counter, deque
from dataclasses import dataclass

from .render import Grid, ascii_grid


@dataclass(frozen=True)
class Component:
    color: int
    size: int
    top: int
    left: int
    bottom: int
    right: int

    @property
    def center_xy(self) -> tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2


def _flood(grid: Grid, seen: set[tuple[int, int]], r: int, c: int) -> Component:
    color, queue, cells = grid[r][c], deque([(r, c)]), []
    seen.add((r, c))
    while queue:
        y, x = queue.popleft()
        cells.append((y, x))
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            inside = 0 <= ny < len(grid) and 0 <= nx < len(grid[0])
            if inside and (ny, nx) not in seen and grid[ny][nx] == color:
                seen.add((ny, nx))
                queue.append((ny, nx))
    rows, cols = [y for y, _ in cells], [x for _, x in cells]
    return Component(color, len(cells), min(rows), min(cols), max(rows), max(cols))


def components(grid: Grid, skip_background: bool = True) -> list[Component]:
    """4-connected same-color regions; the most common color is background."""
    counts = Counter(cell for row in grid for cell in row)
    background = counts.most_common(1)[0][0] if skip_background else -1
    seen: set[tuple[int, int]] = set()
    return [
        _flood(grid, seen, r, c)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell != background and (r, c) not in seen
    ]


def zoom(grid: Grid, top: int, left: int, bottom: int, right: int) -> str:
    """ASCII of the inclusive window, for close inspection."""
    return ascii_grid([row[left : right + 1] for row in grid[top : bottom + 1]])


Signature = tuple[int, int]  # (color, size) of the region under a click


def signature_at(grid: Grid, x: int, y: int) -> Signature:
    region = _flood(grid, set(), y, x)
    return region.color, region.size


def click_candidates(
    grid: Grid,
    limit: int = 8,
    max_size: int = 200,
    exclude: set[Signature] | frozenset[Signature] = frozenset(),
) -> list[tuple[int, int]]:
    """(x, y) centers of small regions, rarest color first: likely buttons.

    Regions whose (color, size) signature is in `exclude` are skipped.
    """
    counts = Counter(cell for row in grid for cell in row)
    small = [
        p
        for p in components(grid)
        if p.size <= max_size and (p.color, p.size) not in exclude
    ]
    small.sort(key=lambda p: (counts[p.color], p.size))
    return [p.center_xy for p in small[:limit]]
