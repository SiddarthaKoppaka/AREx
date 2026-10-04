"""Component appearance, disappearance, and shape-preserving translations.

Purely geometric: a translation is a disappeared component and an appeared
component with the same layer, value, and normalized cell shape. Nothing here
names objects, roles, or causes.
"""

from arc_agi_3.context.frame_components import FrameLike, Points, component_points
from arc_agi_3.contracts.transition import ComponentSummary, ComponentTranslation

Component = tuple[int, int, Points]
MAX_PAIRS = 64


def _summary(component: Component) -> ComponentSummary:
    layer, value, points = component
    return ComponentSummary(
        layer=layer,
        value=value,
        size=len(points),
        min_row=min(point[0] for point in points),
        min_col=min(point[1] for point in points),
        max_row=max(point[0] for point in points),
        max_col=max(point[1] for point in points),
    )


def _origin(points: Points) -> tuple[int, int]:
    return min(point[0] for point in points), min(point[1] for point in points)


def _shape(points: Points) -> frozenset[tuple[int, int]]:
    row, col = _origin(points)
    return frozenset((y - row, x - col) for y, x in points)


def component_changes(
    before: FrameLike, after: FrameLike
) -> tuple[list[ComponentSummary], list[ComponentSummary], list[ComponentTranslation]]:
    """Return (disappeared, appeared, translations), smallest components first."""
    old, new = set(component_points(before)), set(component_points(after))
    gone = sorted(old - new, key=_sort_key)
    came = sorted(new - old, key=_sort_key)
    by_shape: dict[tuple[int, int, frozenset[tuple[int, int]]], list[Component]] = {}
    for component in came:
        layer, value, points = component
        by_shape.setdefault((layer, value, _shape(points)), []).append(component)
    translations: list[ComponentTranslation] = []
    for layer, value, points in gone:
        if len(translations) >= MAX_PAIRS:
            break
        source = _origin(points)
        for _, _, target in by_shape.get((layer, value, _shape(points)), []):
            destination = _origin(target)
            translations.append(
                ComponentTranslation(
                    layer=layer,
                    value=value,
                    size=len(points),
                    from_row=source[0],
                    from_col=source[1],
                    d_row=destination[0] - source[0],
                    d_col=destination[1] - source[1],
                )
            )
            if len(translations) >= MAX_PAIRS:
                break
    return [_summary(c) for c in gone], [_summary(c) for c in came], translations


def _sort_key(component: Component) -> tuple[int, int, int, int]:
    layer, value, points = component
    row, col = _origin(points)
    return len(points), layer, row, col
