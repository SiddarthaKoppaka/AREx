"""Compare each observable prediction claim with one observed transition."""

from pydantic import JsonValue

from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.prediction import ExpectedOutcome, TranslationExpectation
from arc_agi_3.contracts.transition import ComponentTranslation, TransitionEvidence
from arc_agi_3.contracts.verification import ClaimCheck

from .components import component_changes


def _check(
    claim: str, ok: bool | None, expected: object, observed: object
) -> ClaimCheck:
    status = "not_evaluable" if ok is None else ("matched" if ok else "mismatched")
    return ClaimCheck.model_validate(
        {"claim": claim, "status": status, "expected": expected, "observed": observed}
    )


def check_claims(
    expected: ExpectedOutcome,
    before: Observation,
    after: Observation,
    evidence: TransitionEvidence,
) -> tuple[ClaimCheck, ...]:
    checks: list[ClaimCheck] = []
    if expected.observation_hash:
        ok = expected.observation_hash == after.observation_hash
        checks.append(
            _check(
                "observation_hash",
                ok,
                expected.observation_hash,
                after.observation_hash,
            )
        )
    if expected.state:
        checks.append(
            _check("state", expected.state == after.state, expected.state, after.state)
        )
    if expected.min_levels_completed is not None:
        ok = after.levels_completed >= expected.min_levels_completed
        checks.append(
            _check(
                "levels_completed",
                ok,
                expected.min_levels_completed,
                after.levels_completed,
            )
        )
    low, high = expected.min_changed_cells, expected.max_changed_cells
    if low is not None or high is not None:
        count = evidence.changed_cells
        ok = (low is None or count >= low) and (high is None or count <= high)
        checks.append(_check("changed_cells", ok, {"min": low, "max": high}, count))
    if expected.no_op is not None:
        unchanged = evidence.changed_cells == 0 and not evidence.metadata_changes
        checks.append(
            _check("no_op", unchanged == expected.no_op, expected.no_op, unchanged)
        )
    for cell in expected.cell_values:
        value = _cell(after, cell.layer, cell.row, cell.col)
        same = None if value is None else value == cell.value
        checks.append(_check("cell_value", same, cell.model_dump(mode="json"), value))
    if expected.translations:
        moved = list(evidence.translations)
        if evidence.omitted_translations:
            moved = component_changes(before.frame, after.frame)[2]
        checks.extend(_translation(item, moved) for item in expected.translations)
    return tuple(checks)


def _cell(observation: Observation, layer: int, row: int, col: int) -> int | None:
    frame = observation.frame
    if layer >= len(frame) or row >= len(frame[layer]) or col >= len(frame[layer][row]):
        return None
    return frame[layer][row][col]


def _translation(
    item: TranslationExpectation, moved: list[ComponentTranslation]
) -> ClaimCheck:
    same_value = [m for m in moved if m.layer == item.layer and m.value == item.value]
    ok = any(
        m.d_row == item.d_row
        and m.d_col == item.d_col
        and item.from_row in (None, m.from_row)
        and item.from_col in (None, m.from_col)
        for m in same_value
    )
    observed: list[JsonValue] = [
        {"from": [m.from_row, m.from_col], "delta": [m.d_row, m.d_col]}
        for m in same_value[:4]
    ]
    return _check("translation", ok, item.model_dump(mode="json"), observed)
