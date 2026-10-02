"""Rich transition evidence is exact, bounded, and free of semantic labels."""

from arc_agi_3.evidence import build_transition
from arc_agi_3.testing.epistemic import grid
from arc_agi_3.trace.canonical import canonical_json


def blank(size: int = 6) -> list[list[int]]:
    return [[0] * size for _ in range(size)]


def test_zero_cell_transition_has_no_changes() -> None:
    same = grid(blank())
    evidence = build_transition(same, same)
    assert evidence.changed_cells == 0
    assert evidence.before_hash == evidence.after_hash == same.observation_hash
    assert evidence.cell_changes == () and evidence.translations == ()
    assert evidence.changed_regions == () and evidence.value_count_changes == ()


def test_single_cell_change_reports_values_counts_and_region() -> None:
    after = blank()
    after[3][4] = 7
    evidence = build_transition(grid(blank()), grid(after))
    assert evidence.changed_cells == 1
    change = evidence.cell_changes[0]
    assert (change.layer, change.row, change.col) == (0, 3, 4)
    assert (change.before, change.after) == (0, 7)
    assert [(v.value, v.before, v.after) for v in evidence.value_count_changes] == [
        (0, 36, 35),
        (7, 0, 1),
    ]
    region = evidence.changed_regions[0]
    assert (region.min_row, region.min_col, region.max_row, region.max_col) == (
        3,
        4,
        3,
        4,
    )
    assert evidence.appeared_components[0].value == 7
    assert evidence.translations == ()


def test_two_cell_apparent_translation_is_geometric_only() -> None:
    before, after = blank(), blank()
    before[1][1] = 1
    after[1][2] = 1
    evidence = build_transition(grid(before), grid(after))
    assert evidence.changed_cells == 2
    moved = evidence.translations[0]
    assert (moved.value, moved.from_row, moved.from_col) == (1, 1, 1)
    assert (moved.d_row, moved.d_col) == (0, 1)
    text = canonical_json(evidence).lower()
    assert "player" not in text and "moved up" not in text


def test_many_cell_transition_is_summarized_with_explicit_omissions() -> None:
    before = [[0] * 64 for _ in range(64)]
    after = [[(row + col) % 5 for col in range(64)] for row in range(64)]
    evidence = build_transition(grid(before), grid(after))
    assert evidence.changed_cells == sum(
        (row + col) % 5 != 0 for row in range(64) for col in range(64)
    )
    assert len(evidence.cell_changes) == 24
    assert evidence.omitted_cell_changes == evidence.changed_cells - 24
    assert len(evidence.appeared_components) <= 6
    assert evidence.omitted_component_changes > 0
    assert len(canonical_json(evidence)) < 6_000
