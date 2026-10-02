"""Equal-value connected components with no semantic labels."""

from collections.abc import Iterator, Sequence

FrameLike = Sequence[Sequence[Sequence[int]]]
Points = frozenset[tuple[int, int]]


def component_points(frame: FrameLike) -> Iterator[tuple[int, int, Points]]:
    """Yield (layer, value, cells) for every 4-connected equal-value region."""
    for z, layer in enumerate(frame):
        seen: set[tuple[int, int]] = set()
        for y, row in enumerate(layer):
            for x, value in enumerate(row):
                if (y, x) in seen:
                    continue
                queue, points = [(y, x)], []
                seen.add((y, x))
                while queue:
                    cy, cx = queue.pop()
                    points.append((cy, cx))
                    for ny, nx in (
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                    ):
                        if (
                            0 <= ny < len(layer)
                            and 0 <= nx < len(layer[ny])
                            and (ny, nx) not in seen
                            and layer[ny][nx] == value
                        ):
                            seen.add((ny, nx))
                            queue.append((ny, nx))
                yield z, value, frozenset(points)


def bounds(layer: int, value: int, points: Points) -> dict[str, int]:
    return {
        "layer": layer,
        "value": value,
        "size": len(points),
        "min_y": min(point[0] for point in points),
        "min_x": min(point[1] for point in points),
        "max_y": max(point[0] for point in points),
        "max_x": max(point[1] for point in points),
    }


def components(frame: FrameLike) -> list[dict[str, int]]:
    found = [bounds(*item) for item in component_points(frame)]
    return sorted(
        found,
        key=lambda item: (item["size"], item["layer"], item["min_y"], item["min_x"]),
    )
